"""CLI to precompute and cache concept vectors for Llama-3.1-70B-Instruct."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from nnsight import LanguageModel

from concept_vectors import compute_concept_vectors
from prompts import build_concept_prompt_messages, tokenize_concept_prompt

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "meta-llama/Llama-3.1-70B-Instruct"
DEFAULT_CONCEPTS_FILE = ROOT_DIR / "data" / "concepts.txt"
DEFAULT_BASELINE_FILE = ROOT_DIR / "data" / "baseline_words.txt"
DEFAULT_CACHE_PATH = ROOT_DIR / "data" / "vectors" / "concept_vectors.pt"


def parse_word_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [word.strip() for word in re.split(r"[,\n]", text) if word.strip()]


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def parse_layers(layers_arg: str, num_hidden_layers: int) -> list[int]:
    if layers_arg.lower() == "all":
        return list(range(num_hidden_layers))

    out: set[int] = set()
    for chunk in layers_arg.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue

        if "-" in chunk:
            start_text, end_text = chunk.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid layer range '{chunk}': end < start.")
            for layer_idx in range(start, end + 1):
                out.add(layer_idx)
        else:
            out.add(int(chunk))

    layers = sorted(out)
    if not layers:
        raise ValueError("No layers were parsed from --layers.")
    if layers[0] < 0 or layers[-1] >= num_hidden_layers:
        raise ValueError(
            f"Layer indices must be within [0, {num_hidden_layers - 1}], got {layers}."
        )
    return layers


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute and cache concept vectors in data/vectors/concept_vectors.pt "
            "for remote NDIF tracing."
        )
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model id to use with nnsight/NDIF.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional model revision/branch.",
    )
    parser.add_argument(
        "--concepts-file",
        type=Path,
        default=DEFAULT_CONCEPTS_FILE,
        help="File containing concept words (comma and/or newline separated).",
    )
    parser.add_argument(
        "--baseline-file",
        type=Path,
        default=DEFAULT_BASELINE_FILE,
        help="File containing baseline words (comma and/or newline separated).",
    )
    parser.add_argument(
        "--layers",
        default="all",
        help=(
            "Layer selection: 'all' or comma/range list such as '0-7,12,24-31'. "
            "Default: all layers."
        ),
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run locally instead of NDIF remote execution.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="How many times to retry model init / vector extraction on transient failures.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run one scan/validate trace and print output shapes, then exit without "
            "computing vectors."
        ),
    )
    return parser


def run_with_retries(task_name: str, fn, max_retries: int):
    wait_seconds = 15
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if attempt == max_retries:
                raise
            print(
                f"[warn] {task_name} failed (attempt {attempt}/{max_retries}): {exc}\n"
                f"[warn] Retrying in {wait_seconds}s..."
            )
            time.sleep(wait_seconds)
            wait_seconds = min(wait_seconds * 2, 180)


def _layer_envoy(model: LanguageModel, layer_idx: int):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    raise AttributeError("Could not locate transformer layers on this LanguageModel.")


def run_dry_run(
    model: LanguageModel,
    concept_word: str,
    layer_idx: int,
    remote: bool,
) -> None:
    if getattr(model, "tokenizer", None) is None:
        raise RuntimeError("LanguageModel tokenizer is not initialized.")

    messages = build_concept_prompt_messages(concept_word)
    tokenized_prompt = tokenize_concept_prompt(model.tokenizer, messages)
    layer_envoy = _layer_envoy(model, layer_idx)

    hidden = None
    last_token = None
    with model.trace(remote=remote, scan=True, validate=True) as tracer:
        with tracer.invoke(tokenized_prompt):
            layer_hidden = layer_envoy.output[0]
            hidden = layer_hidden.detach().cpu().save()
            last_token = layer_hidden[..., -1, :].squeeze(0).detach().cpu().save()

    if hidden is None or last_token is None:
        raise RuntimeError("Dry-run trace did not return expected tensors.")

    print(f"[dry-run] Prompt: Tell me about {concept_word}.")
    print(f"[dry-run] Layer {layer_idx} hidden shape: {tuple(hidden.shape)}")
    print(f"[dry-run] Layer {layer_idx} last-token shape: {tuple(last_token.shape)}")


def main() -> None:
    args = build_arg_parser().parse_args()
    load_dotenv()

    if not args.concepts_file.exists():
        raise FileNotFoundError(f"Concepts file not found: {args.concepts_file}")
    if not args.baseline_file.exists():
        raise FileNotFoundError(f"Baseline file not found: {args.baseline_file}")

    concepts = parse_word_file(args.concepts_file)
    baseline_words = parse_word_file(args.baseline_file)
    if not concepts:
        raise ValueError(f"No concept words found in {args.concepts_file}.")
    if not baseline_words:
        raise ValueError(f"No baseline words found in {args.baseline_file}.")

    # compute_concept_vectors expects unique concept keys in output dict.
    unique_concepts = unique_preserve_order(concepts)
    removed = len(concepts) - len(unique_concepts)
    if removed > 0:
        print(f"[info] Removed {removed} duplicate concept words before vectorization.")

    print("[info] Initializing nnsight model wrapper...")
    model = run_with_retries(
        task_name="model initialization",
        fn=lambda: LanguageModel(args.model, revision=args.revision),
        max_retries=args.max_retries,
    )

    try:
        num_hidden_layers = len(model.model.layers)
    except Exception as exc:
        raise RuntimeError(
            "Could not infer number of layers from model.model.layers."
        ) from exc

    layers = parse_layers(args.layers, num_hidden_layers)
    remote = not args.local

    print(
        f"[info] Model: {args.model} "
        f"(revision={args.revision or 'main'}, layers={len(layers)}, remote={remote})"
    )
    print(
        f"[info] Concepts: {len(unique_concepts)} | Baselines: {len(baseline_words)}"
    )

    if args.dry_run:
        print("[info] Running dry-run trace (scan=True, validate=True) on one prompt...")
        run_with_retries(
            task_name="dry-run validation",
            fn=lambda: run_dry_run(
                model=model,
                concept_word=unique_concepts[0],
                layer_idx=layers[0],
                remote=remote,
            ),
            max_retries=args.max_retries,
        )
        print("[done] Dry-run completed. Exiting without computing concept vectors.")
        return

    print("[info] Starting vector extraction job...")
    start = time.time()

    concept_vectors = run_with_retries(
        task_name="concept vector extraction",
        fn=lambda: compute_concept_vectors(
            model=model,
            words=unique_concepts,
            baseline_words=baseline_words,
            layers=layers,
            remote=remote,
            logger=lambda message: print(f"[trace] {message}", flush=True),
        ),
        max_retries=args.max_retries,
    )

    elapsed = time.time() - start
    print(
        f"[done] Cached concept vectors to {DEFAULT_CACHE_PATH} "
        f"in {elapsed:.1f}s."
    )
    print(
        "[done] Tensor map shape summary: "
        f"{len(concept_vectors)} layers x {len(unique_concepts)} concepts."
    )


if __name__ == "__main__":
    main()
