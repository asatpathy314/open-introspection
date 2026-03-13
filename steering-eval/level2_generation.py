#!/usr/bin/env python3
"""
Level 2: Open-Ended Generation Evaluation (Rimsky et al., ACL 2024)

Inject a steering vector during generation and evaluate via:
  1. LLM judge (concept identification)
  2. String-match concept mention rate
  3. Coherence judge

Run as: python level2_generation.py [--norm raw] [--layers 40] [--concepts snow dust]
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch

from config import (
    ALPHA_SWEEP,
    CONCEPT_WORDS,
    GENERATION_MAX_TOKENS,
    GENERATION_TEMPERATURE,
    GENERATION_TOP_P,
    MAX_RETRIES,
    RESULTS_DIR,
    RETRY_DELAY,
    SEED,
    VECTOR_DIR,
)
from ndif_utils import setup_ndif
from vectors import get_baseline_norms, load_all_concept_vectors, normalize_vector

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

DEFAULT_NEUTRAL_PROMPT = (
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


def normalize_prompt_label(prompt_label: str) -> str:
    """Generate a filesystem-friendly prompt label."""
    chars = []
    for ch in prompt_label.lower():
        if ch.isalnum():
            chars.append(ch)
        elif ch in {" ", "-", "_"}:
            chars.append("_")
    normalized = "".join(chars).strip("_")
    return normalized or "prompt"


def generate_steered(
    model, prompt: str, layer_idx: int, steering_vec, alpha: float, max_new_tokens: int
) -> str:
    """Generate with steering. MUST be in __main__ file for nnsight."""
    input_ids = model.tokenizer(prompt, return_tensors="pt")["input_ids"]
    prompt_len = input_ids.shape[1]

    if steering_vec is not None and alpha != 0:
        with model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=GENERATION_TEMPERATURE,
            top_p=GENERATION_TOP_P,
            remote=True,
        ):
            hs = model.model.layers[layer_idx].output[0]
            sv = (alpha * steering_vec).to(device=hs.device, dtype=hs.dtype)
            scope = generate_steered.injection_scope
            if scope == "assistant_only":
                # Rimsky-style generation steering applies only to assistant-side tokens.
                injection = torch.zeros_like(hs)
                if injection.ndim == 3:
                    injection[:, prompt_len:, :] = sv
                elif injection.ndim == 2:
                    injection[prompt_len:, :] = sv
                else:
                    raise ValueError(
                        f"Unexpected hidden-state rank at layer {layer_idx}: ndim={injection.ndim}"
                    )
                model.model.layers[layer_idx].output[0] = hs + injection
            elif scope == "all_positions":
                model.model.layers[layer_idx].output[0] = hs + sv
            else:
                raise ValueError(f"Unknown injection scope: {scope}")
            out_ids = model.generator.output.save()
    else:
        with model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=GENERATION_TEMPERATURE,
            top_p=GENERATION_TOP_P,
            remote=True,
        ):
            out_ids = model.generator.output.save()

    generated_ids = out_ids[0][prompt_len:]
    return model.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Level 2: Open-Ended Generation Eval")
    parser.add_argument(
        "--norm", default="raw", choices=["raw", "unit", "norm_matched"]
    )
    parser.add_argument("--layers", type=int, nargs="+", default=[40])
    parser.add_argument("--concepts", type=str, nargs="+", default=None)
    parser.add_argument("--num-generations", type=int, default=20)
    parser.add_argument("--alphas", type=float, nargs="+", default=None)
    parser.add_argument("--prompt-text", type=str, default=DEFAULT_NEUTRAL_PROMPT)
    parser.add_argument("--prompt-label", type=str, default="today")
    parser.add_argument(
        "--injection-scope",
        type=str,
        default="assistant_only",
        choices=["assistant_only", "all_positions"],
    )
    args = parser.parse_args()

    norm = args.norm
    layers = args.layers
    concepts = (
        [c.lower() for c in args.concepts] if args.concepts else CONCEPT_WORDS[:20]
    )
    alphas = args.alphas or ALPHA_SWEEP[norm]
    num_gens = args.num_generations
    prompt_text = args.prompt_text
    prompt_label = normalize_prompt_label(args.prompt_label)
    injection_scope = args.injection_scope

    model = setup_ndif()
    generate_steered.injection_scope = injection_scope
    baseline_norms = get_baseline_norms()
    all_vecs = load_all_concept_vectors(concepts)
    rng = random.Random(SEED)

    output_name = f"level2_generation_{norm}.jsonl"
    suffixes = []
    if prompt_text != DEFAULT_NEUTRAL_PROMPT or prompt_label != "today":
        suffixes.append(prompt_label)
    if injection_scope != "assistant_only":
        suffixes.append(injection_scope)
    if suffixes:
        output_name = f"level2_generation_{norm}_{'_'.join(suffixes)}.jsonl"
    output_path = RESULTS_DIR / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume
    done_keys = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done_keys.add(
                        (
                            r["concept"],
                            r["layer"],
                            r["alpha"],
                            r["gen_idx"],
                            r.get("prompt_label", "today"),
                            r.get("injection_scope", "assistant_only"),
                        )
                    )
        log.info("Resuming: %d generations already done", len(done_keys))

    total = len(concepts) * len(layers) * len(alphas) * num_gens
    log.info(
        "Generation eval: %d concepts x %d layers x %d alphas x %d gens = %d (prompt=%s)",
        len(concepts),
        len(layers),
        len(alphas),
        num_gens,
        total,
        prompt_label,
    )

    with open(output_path, "a") as out_f:
        for concept in concepts:
            concept_vec_all = all_vecs[concept]
            for layer in layers:
                raw_vec = concept_vec_all[layer]
                bl_norm = baseline_norms[layer].item()
                steering_vec = normalize_vector(raw_vec, norm, bl_norm)

                for alpha in alphas:
                    for gen_idx in range(num_gens):
                        key = (
                            concept,
                            layer,
                            alpha,
                            gen_idx,
                            prompt_label,
                            injection_scope,
                        )
                        if key in done_keys:
                            continue

                        # Generate with retry
                        response = None
                        for attempt in range(1, MAX_RETRIES + 1):
                            try:
                                response = generate_steered(
                                    model,
                                    prompt_text,
                                    layer,
                                    steering_vec,
                                    alpha,
                                    max_new_tokens=GENERATION_MAX_TOKENS,
                                )
                                break
                            except Exception as e:
                                log.warning(
                                    "Gen attempt %d/%d failed: %s",
                                    attempt,
                                    MAX_RETRIES,
                                    e,
                                )
                                if attempt < MAX_RETRIES:
                                    time.sleep(RETRY_DELAY * attempt)

                        if response is None:
                            continue

                        # Evaluate
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
                        except Exception as e:
                            log.warning("Judge ID failed: %s", e)
                            id_correct, id_answer = False, "ERROR"

                        try:
                            coh_prompt = JUDGE_COHERENCE_PROMPT.format(
                                response=response
                            )
                            coh_answer = call_llm_judge(coh_prompt).upper().strip()
                            coherent = coh_answer.startswith("YES")
                        except Exception as e:
                            log.warning("Judge coherence failed: %s", e)
                            coherent = True

                        row = {
                            "concept": concept,
                            "layer": layer,
                            "alpha": alpha,
                            "norm_method": norm,
                            "gen_idx": gen_idx,
                            "prompt_label": prompt_label,
                            "prompt_text": prompt_text,
                            "injection_scope": injection_scope,
                            "response": response[:500],
                            "concept_mentioned": mentioned,
                            "id_correct": id_correct,
                            "id_answer": id_answer,
                            "coherent": coherent,
                        }
                        out_f.write(json.dumps(row) + "\n")
                        out_f.flush()

            log.info("  Done: %s", concept)

    log.info("Generation eval complete -> %s", output_path)


if __name__ == "__main__":
    main()
