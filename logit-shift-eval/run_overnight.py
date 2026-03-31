#!/usr/bin/env python3
"""
Launch and supervise a full corrected logit-shift rerun in a fresh results dir.

The child stages are resumable. This supervisor adds:
- a fresh results directory to avoid mixing old and corrected runs
- periodic heartbeat logging
- stall detection and stage restarts
- automatic layer selection after the validation sweep
- post-run analysis and plot generation
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from analysis import analyze_layer_sweep
from config import (
    ALPHAS,
    CONCEPT_WORDS,
    INJECTION_CONDITIONS,
    LAYER_SWEEP,
    N_RANDOM_VECTORS,
    NEUTRAL_PROMPTS,
    TOKEN_SET_FAMILY_DEFAULT,
    VALIDATION_CONCEPTS,
    VALIDATION_PROMPTS,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_RESULTS = ROOT / "data" / "results" / "logit-shift-eval"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


@dataclass
class StageSpec:
    name: str
    cmd: list[str]
    log_path: Path
    output_path: Path | None = None
    expected_rows: int | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    tmp.replace(path)


def append_line(path: Path, message: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(message.rstrip() + "\n")


def count_lines(path: Path) -> int | None:
    if not path.exists() or not path.is_file():
        return None
    with open(path) as f:
        return sum(1 for _ in f)


def file_bytes(path: Path) -> int | None:
    if not path.exists():
        return None
    return path.stat().st_size


def latest_mtime(paths: list[Path]) -> float | None:
    mtimes = [p.stat().st_mtime for p in paths if p.exists()]
    return max(mtimes) if mtimes else None


def stage_is_complete(stage: StageSpec) -> bool:
    if not stage.output_path or not stage.output_path.exists():
        return False
    if stage.output_path.suffix == ".jsonl" and stage.expected_rows is not None:
        lines = count_lines(stage.output_path)
        return lines is not None and lines >= stage.expected_rows
    size = file_bytes(stage.output_path)
    return size is not None and size > 0


def ensure_unembed(results_dir: Path, source_unembed: Path):
    target = results_dir / "unembed.pt"
    if target.exists():
        log.info("Using existing unembed at %s", target)
        return
    if not source_unembed.exists():
        raise FileNotFoundError(f"Source unembed not found: {source_unembed}")

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source_unembed, target)
        log.info("Hard-linked unembed.pt into %s", target)
        return
    except OSError:
        pass

    try:
        target.symlink_to(source_unembed.resolve())
        log.info("Symlinked unembed.pt into %s", target)
        return
    except OSError:
        pass

    shutil.copy2(source_unembed, target)
    log.info("Copied unembed.pt into %s", target)


def expected_row_counts() -> dict[str, int]:
    n_nonzero = len([a for a in ALPHAS if a != 0])
    return {
        "layer-sweep": (
            len(VALIDATION_PROMPTS) * len(VALIDATION_CONCEPTS) * len(INJECTION_CONDITIONS)
            + len(VALIDATION_CONCEPTS) * len(VALIDATION_PROMPTS) * n_nonzero
            * len(INJECTION_CONDITIONS) * len(LAYER_SWEEP)
        ),
        "main-sweep": (
            len(NEUTRAL_PROMPTS) * len(CONCEPT_WORDS) * len(INJECTION_CONDITIONS)
            + len(CONCEPT_WORDS) * len(NEUTRAL_PROMPTS) * n_nonzero
            * len(INJECTION_CONDITIONS)
        ),
        "random-sweep": (
            N_RANDOM_VECTORS * len(NEUTRAL_PROMPTS) * n_nonzero * len(INJECTION_CONDITIONS)
        ),
    }


def choose_best_layer(results_dir: Path) -> int:
    layer_path = results_dir / "layer_sweep.jsonl"
    result = analyze_layer_sweep(layer_path)
    best = result.get("best", {})
    chosen = best.get("all_positions") or best.get("last_token")
    if not chosen:
        raise RuntimeError(f"Could not select best layer from {layer_path}")

    payload = {
        "chosen_layer": int(chosen["layer"]),
        "chosen_injection_basis": chosen["injection"],
        "best_by_injection": best,
        "computed_at_utc": utc_now(),
    }
    atomic_write_json(results_dir / "selected_layer.json", payload)
    return int(chosen["layer"])


def terminate_process_group(proc: subprocess.Popen, timeout_seconds: int = 30):
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(1)

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def stage_status_payload(
    stage: StageSpec,
    attempt: int,
    proc: subprocess.Popen | None,
    results_dir: Path,
    started_at: float,
    state: str,
    note: str = "",
) -> dict:
    payload = {
        "updated_at_utc": utc_now(),
        "results_dir": str(results_dir),
        "stage": stage.name,
        "attempt": attempt,
        "state": state,
        "pid": proc.pid if proc else None,
        "started_at_unix": started_at,
        "started_at_utc": datetime.fromtimestamp(started_at, timezone.utc).isoformat(),
        "note": note,
        "log_path": str(stage.log_path),
        "output_path": str(stage.output_path) if stage.output_path else None,
    }

    if stage.output_path:
        payload["output_bytes"] = file_bytes(stage.output_path)
        if stage.output_path.suffix == ".jsonl":
            payload["output_lines"] = count_lines(stage.output_path)
            if stage.expected_rows:
                payload["expected_rows"] = stage.expected_rows
                lines = payload["output_lines"] or 0
                payload["progress_fraction"] = min(lines / stage.expected_rows, 1.0)
    return payload


def run_stage(
    stage: StageSpec,
    *,
    results_dir: Path,
    env: dict[str, str],
    poll_seconds: int,
    stall_seconds: int,
    max_restarts: int,
    supervisor_log: Path,
    status_path: Path,
):
    for attempt in range(1, max_restarts + 1):
        stage.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stage.log_path, "a") as stage_log:
            stage_log.write(
                f"\n[{utc_now()}] starting stage={stage.name} attempt={attempt}\n"
            )
            stage_log.flush()
            proc = subprocess.Popen(
                stage.cmd,
                cwd=ROOT,
                env=env,
                stdout=stage_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        started_at = time.time()
        append_line(
            supervisor_log,
            f"[{utc_now()}] stage={stage.name} attempt={attempt} pid={proc.pid} started",
        )
        atomic_write_json(
            status_path,
            stage_status_payload(
                stage, attempt, proc, results_dir, started_at, "running"
            ),
        )

        progress_paths = [stage.log_path]
        if stage.output_path:
            progress_paths.append(stage.output_path)
        last_progress_mtime = latest_mtime(progress_paths) or started_at

        while True:
            rc = proc.poll()
            current_mtime = latest_mtime(progress_paths) or last_progress_mtime
            if current_mtime > last_progress_mtime:
                last_progress_mtime = current_mtime

            if rc is not None:
                note = f"exit_code={rc}"
                append_line(
                    supervisor_log,
                    f"[{utc_now()}] stage={stage.name} attempt={attempt} exited rc={rc}",
                )
                atomic_write_json(
                    status_path,
                    stage_status_payload(
                        stage,
                        attempt,
                        proc,
                        results_dir,
                        started_at,
                        "completed" if rc == 0 else "failed",
                        note=note,
                    ),
                )
                if rc == 0:
                    return
                break

            idle_seconds = int(time.time() - last_progress_mtime)
            note = f"idle_seconds={idle_seconds}"
            append_line(
                supervisor_log,
                f"[{utc_now()}] heartbeat stage={stage.name} attempt={attempt} pid={proc.pid} "
                f"idle_seconds={idle_seconds}",
            )
            atomic_write_json(
                status_path,
                stage_status_payload(
                    stage, attempt, proc, results_dir, started_at, "running", note=note
                ),
            )

            if idle_seconds >= stall_seconds:
                append_line(
                    supervisor_log,
                    f"[{utc_now()}] stage={stage.name} attempt={attempt} stalled; terminating",
                )
                terminate_process_group(proc)
                atomic_write_json(
                    status_path,
                    stage_status_payload(
                        stage,
                        attempt,
                        proc,
                        results_dir,
                        started_at,
                        "restarting",
                        note=f"stalled_after_seconds={idle_seconds}",
                    ),
                )
                break

            time.sleep(poll_seconds)

        if attempt < max_restarts:
            append_line(
                supervisor_log,
                f"[{utc_now()}] stage={stage.name} scheduling retry {attempt + 1}/{max_restarts}",
            )
            time.sleep(min(60, poll_seconds))

    raise RuntimeError(f"Stage failed after {max_restarts} attempts: {stage.name}")


def build_stage_specs(
    results_dir: Path,
    chosen_layer: int | None = None,
    token_family: str = TOKEN_SET_FAMILY_DEFAULT,
) -> list[StageSpec]:
    counts = expected_row_counts()
    logs_dir = results_dir / "logs"

    stages = [
        StageSpec(
            name="compute-token-sets",
            cmd=["uv", "run", "python", "-u", "logit-shift-eval/run_sweep.py", "compute-token-sets"],
            log_path=logs_dir / "compute-token-sets.log",
            output_path=results_dir / "token_sets.json",
        ),
        StageSpec(
            name="layer-sweep",
            cmd=[
                "uv", "run", "python", "-u", "logit-shift-eval/run_sweep.py",
                "layer-sweep", "--token-family", token_family,
            ],
            log_path=logs_dir / "layer-sweep.log",
            output_path=results_dir / "layer_sweep.jsonl",
            expected_rows=counts["layer-sweep"],
        ),
    ]

    if chosen_layer is not None:
        stages.extend(
            [
                StageSpec(
                    name="main-sweep",
                    cmd=[
                        "uv", "run", "python", "-u", "logit-shift-eval/run_sweep.py",
                        "main-sweep", "--layer", str(chosen_layer),
                        "--token-family", token_family,
                    ],
                    log_path=logs_dir / "main-sweep.log",
                    output_path=results_dir / f"main_sweep_layer{chosen_layer}.jsonl",
                    expected_rows=counts["main-sweep"],
                ),
                StageSpec(
                    name="random-sweep",
                    cmd=[
                        "uv", "run", "python", "-u", "logit-shift-eval/run_sweep.py",
                        "random-sweep", "--layer", str(chosen_layer),
                        "--token-family", token_family,
                    ],
                    log_path=logs_dir / "random-sweep.log",
                    output_path=results_dir / f"random_sweep_layer{chosen_layer}.jsonl",
                    expected_rows=counts["random-sweep"],
                ),
                StageSpec(
                    name="analysis",
                    cmd=[
                        "uv", "run", "python", "-u", "logit-shift-eval/analysis.py",
                        "--layer", str(chosen_layer), "--token-family", token_family,
                    ],
                    log_path=logs_dir / "analysis.log",
                    output_path=results_dir / f"analysis_layer{chosen_layer}.json",
                ),
                StageSpec(
                    name="plots",
                    cmd=[
                        "uv", "run", "python", "-u", "logit-shift-eval/plots.py",
                        "--layer", str(chosen_layer), "--token-family", token_family,
                    ],
                    log_path=logs_dir / "plots.log",
                    output_path=results_dir / "figures_combined.pdf",
                ),
            ]
        )

    return stages


def main():
    parser = argparse.ArgumentParser(description="Overnight supervisor for logit-shift rerun")
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Fresh results directory for the corrected rerun",
    )
    parser.add_argument(
        "--source-unembed",
        type=Path,
        default=DEFAULT_SOURCE_RESULTS / "unembed.pt",
        help="Existing unembed.pt to reuse",
    )
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--stall-seconds", type=int, default=1800)
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--token-family", type=str, default=TOKEN_SET_FAMILY_DEFAULT)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    if not os.environ.get("NDIF_API_KEY"):
        raise RuntimeError("NDIF_API_KEY not available after loading .env")

    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    ensure_unembed(results_dir, args.source_unembed.resolve())

    supervisor_log = results_dir / "overnight_supervisor.log"
    status_path = results_dir / "overnight_status.json"
    manifest = {
        "started_at_utc": utc_now(),
        "results_dir": str(results_dir),
        "source_unembed": str(args.source_unembed.resolve()),
        "poll_seconds": args.poll_seconds,
        "stall_seconds": args.stall_seconds,
        "max_restarts": args.max_restarts,
        "token_family": args.token_family,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    atomic_write_json(results_dir / "run_manifest.json", manifest)
    append_line(
        supervisor_log,
        f"[{utc_now()}] supervisor started results_dir={results_dir}",
    )

    env = os.environ.copy()
    env["RESULTS_DIR"] = str(results_dir)
    env["TOKEN_SET_FAMILY"] = args.token_family
    env["PYTHONUNBUFFERED"] = "1"

    try:
        for stage in build_stage_specs(results_dir, token_family=args.token_family):
            if stage_is_complete(stage):
                append_line(
                    supervisor_log,
                    f"[{utc_now()}] skipping completed stage={stage.name}",
                )
                atomic_write_json(
                    status_path,
                    {
                        "updated_at_utc": utc_now(),
                        "results_dir": str(results_dir),
                        "stage": stage.name,
                        "state": "skipped_completed",
                        "output_path": str(stage.output_path) if stage.output_path else None,
                    },
                )
                continue
            run_stage(
                stage,
                results_dir=results_dir,
                env=env,
                poll_seconds=args.poll_seconds,
                stall_seconds=args.stall_seconds,
                max_restarts=args.max_restarts,
                supervisor_log=supervisor_log,
                status_path=status_path,
            )

        chosen_layer = choose_best_layer(results_dir)
        append_line(
            supervisor_log,
            f"[{utc_now()}] selected best layer={chosen_layer}",
        )

        for stage in build_stage_specs(
            results_dir, chosen_layer=chosen_layer, token_family=args.token_family
        )[2:]:
            if stage_is_complete(stage):
                append_line(
                    supervisor_log,
                    f"[{utc_now()}] skipping completed stage={stage.name}",
                )
                atomic_write_json(
                    status_path,
                    {
                        "updated_at_utc": utc_now(),
                        "results_dir": str(results_dir),
                        "stage": stage.name,
                        "state": "skipped_completed",
                        "output_path": str(stage.output_path) if stage.output_path else None,
                    },
                )
                continue
            run_stage(
                stage,
                results_dir=results_dir,
                env=env,
                poll_seconds=args.poll_seconds,
                stall_seconds=args.stall_seconds,
                max_restarts=args.max_restarts,
                supervisor_log=supervisor_log,
                status_path=status_path,
            )

        atomic_write_json(
            status_path,
            {
                "updated_at_utc": utc_now(),
                "results_dir": str(results_dir),
                "state": "finished",
                "selected_layer": chosen_layer,
            },
        )
        append_line(
            supervisor_log,
            f"[{utc_now()}] supervisor finished selected_layer={chosen_layer}",
        )
    except Exception as e:
        atomic_write_json(
            status_path,
            {
                "updated_at_utc": utc_now(),
                "results_dir": str(results_dir),
                "state": "failed",
                "error": str(e),
            },
        )
        append_line(supervisor_log, f"[{utc_now()}] supervisor failed error={e}")
        raise


if __name__ == "__main__":
    main()
