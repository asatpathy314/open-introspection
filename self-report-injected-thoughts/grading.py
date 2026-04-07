"""Grade injected-thought experiment trials with Anthropic batch judging.

This module is specific to the main injected-thought experiment in this folder.
It expects one or more experiment output directories that each contain a
``trials.jsonl`` file produced by ``run_trials.py``.

Trial records fall into two categories:

1. Injection trials, where ``concept`` is a real word and all four rubrics apply.
2. Control trials, where ``concept`` is ``None`` and only the coherence and
   affirmative-response rubrics make sense.

The grader submits one Anthropic batch request per rubric, writes graded
injection and control trials to separate files, and stores the prompt templates
and batch metadata alongside the analysis output.
"""

import json
import time
from pathlib import Path
from typing import Any

import anthropic
import prompt
from anthropic.types.messages.batch_create_params import Request
from dotenv import load_dotenv

MODEL = "claude-sonnet-4-6"
RUBRIC_NAMES = [
    "coherence",
    "thinking_about_word",
    "affirmative_response",
    "affirmative_with_identification",
]
CONTROL_EXCLUDED_RUBRICS = {
    "thinking_about_word",
    "affirmative_with_identification",
}
PROMPTS_BY_RUBRIC = {
    "coherence": prompt.COHERENCE_PROMPT,
    "thinking_about_word": prompt.THINKING_ABOUT_WORD_PROMPT,
    "affirmative_response": prompt.AFFIRMATIVE_RESPONSE_PROMPT,
    "affirmative_with_identification": prompt.AFFIRMATIVE_WITH_IDENTIFICATION_PROMPT,
}
ANALYSIS_DIR = Path(__file__).resolve().parent / "analysis"

load_dotenv()
client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY implicitly


def is_control_trial(trial: dict[str, Any]) -> bool:
    """Return whether a trial is a no-injection control."""
    return trial.get("concept") is None


def rubric_applies_to_trial(rubric_name: str, trial: dict[str, Any]) -> bool:
    """Return whether a rubric should be run for the given trial."""
    return not (is_control_trial(trial) and rubric_name in CONTROL_EXCLUDED_RUBRICS)


def split_trials(
    trials: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split graded trials into injection and control subsets."""
    injection_trials = [trial for trial in trials if not is_control_trial(trial)]
    control_trials = [trial for trial in trials if is_control_trial(trial)]
    return injection_trials, control_trials


def load_trials(experiment_dirs: list[Path]) -> list[dict[str, Any]]:
    """Load and combine ``trials.jsonl`` files from one or more experiment runs."""
    trials = []
    for d in experiment_dirs:
        trials_path = d / "trials.jsonl"
        with trials_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record["experiment_dir"] = str(d)
                trials.append(record)
    return trials


def build_batch_requests(
    trials: list[dict[str, Any]],
    prompt_template: str,
    rubric_name: str,
) -> list[Request]:
    """Build batch requests for a single rubric.

    The prompt template must accept ``{question}``, ``{response}``, and
    ``{word}`` placeholders. Control trials are filtered out for rubrics that
    depend on identifying a specific injected word.
    """
    question = "\n\n".join(
        f"Human: {m['content']}"
        if m["role"] == "user"
        else f"Assistant: {m['content']}"
        for m in prompt.EXPERIMENT_MESSAGES
    )
    requests: list[Request] = []
    for idx, trial in enumerate(trials):
        if not rubric_applies_to_trial(rubric_name, trial):
            continue

        requests.append(
            {
                "custom_id": str(idx),
                "params": {
                    "max_tokens": 256,
                    "temperature": 0,
                    "messages": [
                        {
                            "content": prompt_template.format(
                                question=question,
                                response=trial["response"],
                                word=trial.get("concept") or "",
                            ),
                            "role": "user",
                        }
                    ],
                    "model": MODEL,
                },
            }
        )
    return requests


def grade_trials(
    trials: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Submit grading batches, wait for completion, and attach rubric labels."""
    batches_by_rubric: dict[str, Any] = {}

    for rubric_name in RUBRIC_NAMES:
        batch_requests = build_batch_requests(
            trials,
            PROMPTS_BY_RUBRIC[rubric_name],
            rubric_name,
        )
        if not batch_requests:
            print(f"Skipping batch for {rubric_name}: no applicable trials")
            continue

        batch = client.messages.batches.create(requests=batch_requests)
        batches_by_rubric[rubric_name] = batch
        print(f"Created batch {batch.id} for {rubric_name}")

    # Poll until all batches are done
    pending = set(batches_by_rubric)
    time_start = time.time()
    while pending:
        time.sleep(30)
        for rubric_name in list(pending):
            batches_by_rubric[rubric_name] = client.messages.batches.retrieve(
                batches_by_rubric[rubric_name].id
            )
            if batches_by_rubric[rubric_name].processing_status == "ended":
                print(
                    f"Batch {batches_by_rubric[rubric_name].id} ({rubric_name}) ended"
                )
                pending.discard(rubric_name)
        print(f"Pending: {len(pending)}")

    print(f"All batches ended in {time.time() - time_start:.1f}s")

    # Collect results
    for rubric_name, batch in batches_by_rubric.items():
        for result in client.messages.batches.results(batch.id):
            trial_idx = int(result.custom_id)
            if result.result.type != "succeeded":
                trials[trial_idx][rubric_name] = None
                continue

            text = "".join(
                block.text
                for block in result.result.message.content
                if block.type == "text"
            ).strip()
            if not text:
                trials[trial_idx][rubric_name] = None
                continue

            trials[trial_idx][f"{rubric_name}_raw"] = text
            last_word = text.split()[-1].strip(".,!?:;").upper()
            trials[trial_idx][rubric_name] = (
                last_word if last_word in ("YES", "NO") else None
            )

    return trials, {
        rubric_name: batch.id for rubric_name, batch in batches_by_rubric.items()
    }


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Write records to disk as newline-delimited JSON."""
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def save_analysis(
    trials: list[dict[str, Any]],
    batch_ids: dict[str, str],
    output_dir: Path,
) -> Path:
    """Persist graded outputs plus metadata for reproducibility."""
    output_dir.mkdir(parents=True, exist_ok=True)
    injection_trials, control_trials = split_trials(trials)

    # Save graded trials in separate files so control runs do not have to share
    # a schema with injection trials.
    write_jsonl(injection_trials, output_dir / "graded_trials.jsonl")
    write_jsonl(control_trials, output_dir / "graded_control_trials.jsonl")

    # Save metadata
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(exist_ok=True)

    (metadata_dir / "grading_model.txt").write_text(MODEL)
    (metadata_dir / "batch_ids.json").write_text(json.dumps(batch_ids, indent=2))

    # Save the grading prompts used
    prompts_dir = metadata_dir / "prompts"
    prompts_dir.mkdir(exist_ok=True)
    for name, template in PROMPTS_BY_RUBRIC.items():
        (prompts_dir / f"{name}.txt").write_text(template)

    return output_dir


def print_summary(trials: list[dict[str, Any]], label: str) -> None:
    """Print a compact per-rubric summary for a homogeneous trial subset."""
    print(f"\n{label} ({len(trials)} trials):")
    for rubric in RUBRIC_NAMES:
        applicable_trials = [
            trial for trial in trials if rubric_applies_to_trial(rubric, trial)
        ]
        if not applicable_trials:
            continue

        graded = sum(1 for trial in applicable_trials if trial.get(rubric) is not None)
        yes = sum(1 for trial in applicable_trials if trial.get(rubric) == "YES")
        no = sum(1 for trial in applicable_trials if trial.get(rubric) == "NO")
        failed = len(applicable_trials) - graded
        print(f"  {rubric}: {yes} YES / {no} NO / {failed} failed")


if __name__ == "__main__":
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(
        description=(
            "Grade one or more main-experiment result directories produced by run_trials.py."
        )
    )
    parser.add_argument(
        "experiment_dirs",
        nargs="+",
        type=Path,
        help="Paths to experiment result directories containing trials.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for analysis (default: analysis/<timestamp>)",
    )
    args = parser.parse_args()

    print(f"Loading trials from {len(args.experiment_dirs)} experiment(s)...")
    trials = load_trials(args.experiment_dirs)
    print(f"Loaded {len(trials)} trials")

    trials, batch_ids = grade_trials(trials)

    output_dir = args.output or ANALYSIS_DIR / datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )
    save_analysis(trials, batch_ids, output_dir)
    injection_trials, control_trials = split_trials(trials)

    print(f"\nResults saved to {output_dir}")
    print_summary(injection_trials, "Injection summary")
    if control_trials:
        print_summary(control_trials, "Control summary")
