#!/usr/bin/env python3
"""
Prompt-tail window steering experiment.

This sandbox script keeps the tracked steering harness unchanged. It measures
how much prompt-side coverage is needed to recover the open-ended steering
effect by injecting into only the last K prompt tokens before generation.
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STEERING_EVAL_DIR = PROJECT_ROOT / "steering-eval"
sys.path.insert(0, str(STEERING_EVAL_DIR))

from config import (  # noqa: E402
    CONCEPT_WORDS,
    GENERATION_MAX_TOKENS,
    GENERATION_TEMPERATURE,
    GENERATION_TOP_P,
    MAX_RETRIES,
    RETRY_DELAY,
    SEED,
)
from ndif_utils import setup_ndif  # noqa: E402
from vectors import get_baseline_norms, load_all_concept_vectors, normalize_vector  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "Tell me about today.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)

JUDGE_ID_PROMPT = """Below is a response from an AI model. The model may or may not have been steered to think about a particular concept.

<RESPONSE>
{response}
</RESPONSE>

Which of the following words is most related to the content of this response? Choose exactly one.

{word_list}

Answer with just the word."""

JUDGE_COHERENCE_PROMPT = """Does the following text coherently express a thought? Answer YES or NO.
If the text is garbled, repetitive, or nonsensical, answer NO.

<TEXT>
{response}
</TEXT>"""


def call_llm_judge(prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=50,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def build_injection(hidden_states, steering_delta, prompt_len: int, mode: str):
    injection = torch.zeros_like(hidden_states)

    if mode == "assistant_only":
        if injection.ndim == 3:
            injection[:, prompt_len:, :] = steering_delta
        else:
            injection[prompt_len:, :] = steering_delta
        return injection

    if mode == "all_positions":
        injection += steering_delta
        return injection

    if mode == "prompt_all":
        start = 0
        end = prompt_len
    elif mode.startswith("tail_"):
        window = int(mode.split("_", 1)[1])
        start = max(0, prompt_len - window)
        end = prompt_len
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if injection.ndim == 3:
        injection[:, start:end, :] = steering_delta
    else:
        injection[start:end, :] = steering_delta
    return injection


def generate_with_mode(
    model, prompt: str, layer_idx: int, steering_vec, alpha: float, mode: str
) -> str:
    input_ids = model.tokenizer(prompt, return_tensors="pt")["input_ids"]
    prompt_len = input_ids.shape[1]

    if steering_vec is not None and alpha != 0:
        with model.generate(
            prompt,
            max_new_tokens=GENERATION_MAX_TOKENS,
            do_sample=True,
            temperature=GENERATION_TEMPERATURE,
            top_p=GENERATION_TOP_P,
            remote=True,
        ):
            hs = model.model.layers[layer_idx].output[0]
            steering_delta = (alpha * steering_vec).to(device=hs.device, dtype=hs.dtype)
            injection = build_injection(hs, steering_delta, prompt_len, mode)
            model.model.layers[layer_idx].output[0] = hs + injection
            out_ids = model.generator.output.save()
    else:
        with model.generate(
            prompt,
            max_new_tokens=GENERATION_MAX_TOKENS,
            do_sample=True,
            temperature=GENERATION_TEMPERATURE,
            top_p=GENERATION_TOP_P,
            remote=True,
        ):
            out_ids = model.generator.output.save()

    generated_ids = out_ids[0][prompt_len:]
    return model.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(
        description="Prompt-tail window steering experiment"
    )
    parser.add_argument(
        "--norm", default="raw", choices=["raw", "unit", "norm_matched"]
    )
    parser.add_argument("--layer", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=8)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=[
            "baseline",
            "assistant_only",
            "tail_1",
            "tail_2",
            "tail_4",
            "tail_8",
            "prompt_all",
            "all_positions",
        ],
    )
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--prompt-text", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--concepts", type=str, nargs="+", default=None)
    parser.add_argument(
        "--results-file",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results"
        / "prompt_tail_window.jsonl",
    )
    args = parser.parse_args()

    results_file = args.results_file
    results_file.parent.mkdir(parents=True, exist_ok=True)

    concepts = (
        [c.lower() for c in args.concepts] if args.concepts else CONCEPT_WORDS[:10]
    )
    model = setup_ndif()
    baseline_norms = get_baseline_norms()
    all_vecs = load_all_concept_vectors(concepts)
    rng = random.Random(SEED)

    done_keys = set()
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    done_keys.add((row["concept"], row["mode"], row["gen_idx"]))
        log.info("Resuming: %d rows already present", len(done_keys))

    log.info(
        "Prompt-tail experiment: %d concepts x %d modes x %d gens = %d",
        len(concepts),
        len(args.modes),
        args.num_generations,
        len(concepts) * len(args.modes) * args.num_generations,
    )

    with open(results_file, "a") as out_f:
        for concept in concepts:
            raw_vec = all_vecs[concept][args.layer]
            bl_norm = baseline_norms[args.layer].item()
            steering_vec = normalize_vector(raw_vec, args.norm, bl_norm)

            for mode in args.modes:
                for gen_idx in range(args.num_generations):
                    key = (concept, mode, gen_idx)
                    if key in done_keys:
                        continue

                    alpha = 0.0 if mode == "baseline" else args.alpha
                    response = None
                    for attempt in range(1, MAX_RETRIES + 1):
                        try:
                            actual_mode = (
                                "assistant_only" if mode == "baseline" else mode
                            )
                            response = generate_with_mode(
                                model,
                                args.prompt_text,
                                args.layer,
                                steering_vec,
                                alpha,
                                actual_mode,
                            )
                            break
                        except Exception as exc:
                            log.warning(
                                "Generation failed (concept=%s mode=%s gen=%s attempt=%s/%s): %s",
                                concept,
                                mode,
                                gen_idx,
                                attempt,
                                MAX_RETRIES,
                                exc,
                            )
                            if attempt < MAX_RETRIES:
                                time.sleep(RETRY_DELAY * attempt)

                    if response is None:
                        continue

                    mentioned = concept.lower() in response.lower()

                    try:
                        distractors = [c for c in CONCEPT_WORDS if c != concept]
                        chosen = rng.sample(distractors, min(9, len(distractors)))
                        word_list = chosen + [concept]
                        rng.shuffle(word_list)
                        judge_prompt = JUDGE_ID_PROMPT.format(
                            response=response,
                            word_list="\n".join(
                                f"- {w.capitalize()}" for w in word_list
                            ),
                        )
                        id_answer = (
                            call_llm_judge(judge_prompt).lower().strip().rstrip(".")
                        )
                        id_correct = id_answer == concept.lower()
                    except Exception as exc:
                        log.warning("Judge ID failed for %s/%s: %s", concept, mode, exc)
                        id_answer = "ERROR"
                        id_correct = False

                    try:
                        coh_prompt = JUDGE_COHERENCE_PROMPT.format(response=response)
                        coh_answer = call_llm_judge(coh_prompt).upper().strip()
                        coherent = coh_answer.startswith("YES")
                    except Exception as exc:
                        log.warning(
                            "Judge coherence failed for %s/%s: %s", concept, mode, exc
                        )
                        coherent = True

                    row = {
                        "concept": concept,
                        "layer": args.layer,
                        "norm_method": args.norm,
                        "mode": mode,
                        "alpha": alpha,
                        "gen_idx": gen_idx,
                        "prompt_text": args.prompt_text,
                        "response": response[:500],
                        "concept_mentioned": mentioned,
                        "id_correct": id_correct,
                        "id_answer": id_answer,
                        "coherent": coherent,
                    }
                    out_f.write(json.dumps(row) + "\n")
                    out_f.flush()

            log.info("Done: %s", concept)

    log.info("Prompt-tail experiment complete -> %s", results_file)


if __name__ == "__main__":
    main()
