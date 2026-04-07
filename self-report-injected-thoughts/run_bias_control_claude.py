"""Phase 3: Yes/no bias control experiment.

Injects concept vectors while asking unrelated yes/no questions (where the
correct answer is "no"). If the affirmative rate increases on these unrelated
questions under injection, that suggests the intervention biases the model
toward "yes" rather than enabling genuine anomaly detection.

Structure: 20 questions × 2 layers × 4 alphas = 160 injection + 20 control = 180 trials. AS A NOTE, CLAUDE BUILT THIS.

THE FINAL REPO DOES NOT INCLUDE IT FOR A REASON, IT WAS USED AS AN INTERMEDIARY RESULT DURING RESEARCH
AND REBUILT FOR THE FINAL PAPER.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import nnsight
import torch
from dotenv import load_dotenv
from prompt import CONTROL_PROMPTS

MODEL_ID = "meta-llama/Llama-3.1-70B-Instruct"
VECTOR_DIR = (
    Path(__file__).resolve().parent.parent / "data/vectors/llama-3.1-70b-instruct"
)
NUM_LAYERS = 80
RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass
class BiasControlConfig:
    layers: list[int]
    alphas: list[float]
    questions: list[str]
    concepts: list[str]
    n_control_per_question: int = 1
    concept_seed: int = 42
    max_new_tokens: int = 50
    temperature: float = 1.0
    do_sample: bool = True
    remote: bool = True
    model_id: str = MODEL_ID
    vector_dir: str = str(VECTOR_DIR)


def list_concepts(vector_dir: Path) -> list[str]:
    return sorted(
        p.stem.replace("_all_layers", "") for p in vector_dir.glob("*_all_layers.pt")
    )


def load_concept_vector(vector_dir: Path, concept: str, layer: int) -> torch.Tensor:
    path = vector_dir / f"{concept}_all_layers.pt"
    all_layers = torch.load(path, weights_only=True)
    return all_layers[layer]


def generate_with_injection(
    model: nnsight.LanguageModel,
    input_ids: torch.Tensor,
    vec: torch.Tensor,
    layer: int,
    alpha: float,
    config: BiasControlConfig,
) -> torch.Tensor:
    with model.generate(
        max_new_tokens=config.max_new_tokens,
        do_sample=config.do_sample,
        temperature=config.temperature,
        remote=config.remote,
    ) as tracer:
        with tracer.invoke(input_ids):
            # Inject across all prefill tokens
            hs = model.model.layers[layer].output[0]
            intervention = torch.zeros_like(hs)
            intervention[:, :] = vec * alpha
            model.model.layers[layer].output[0] = hs + intervention
            scaled = alpha * vec.to(device=hs.device, dtype=hs.dtype)

            for _ in tracer.iter[1:]:
                hs = model.model.layers[layer].output[0]
                model.model.layers[layer].output[0] = hs + scaled

        with tracer.invoke():
            output = tracer.result.save()
    return output


def generate_control(
    model: nnsight.LanguageModel,
    input_ids: torch.Tensor,
    config: BiasControlConfig,
) -> torch.Tensor:
    with model.generate(
        max_new_tokens=config.max_new_tokens,
        do_sample=config.do_sample,
        temperature=config.temperature,
        remote=config.remote,
    ) as tracer:
        with tracer.invoke(input_ids):
            output = tracer.result.save()
    return output


def build_question_prompt(tokenizer, question: str) -> tuple[str, torch.Tensor]:
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    return prompt, input_ids


def append_record(path: Path, record: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def load_completed(trials_path: Path) -> set:
    completed = set()
    if not trials_path.exists():
        return completed
    with trials_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # Key: (question_idx, layer, alpha, concept)
            completed.add(
                (
                    r["question_idx"],
                    r.get("layer_idx"),
                    r.get("alpha"),
                    r.get("concept"),
                )
            )
    return completed


def run_experiment(config: BiasControlConfig, output_dir: Path | None = None) -> Path:
    resuming = output_dir is not None and output_dir.exists()
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = RESULTS_DIR / f"bias-control_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)
    trials_path = output_dir / "trials.jsonl"
    completed = load_completed(trials_path)

    if resuming and completed:
        print(f"Resuming into {output_dir} — skipping {len(completed)} trials.")

    model = nnsight.LanguageModel(config.model_id)
    tokenizer = model.tokenizer

    if not resuming:
        config_dict = asdict(config)
        config_dict["timestamp"] = datetime.now().isoformat(timespec="seconds")
        config_dict["python"] = sys.version
        (output_dir / "run_config.json").write_text(json.dumps(config_dict, indent=2))
        result = subprocess.run(["uv", "pip", "freeze"], capture_output=True, text=True)
        (output_dir / "environment.txt").write_text(
            result.stdout if result.returncode == 0 else result.stderr
        )

    print(f"Output dir: {output_dir}")

    rng = random.Random(config.concept_seed)

    # Injection trials
    for q_idx, question in enumerate(config.questions):
        _, input_ids = build_question_prompt(tokenizer, question)
        seq_len = input_ids.shape[1]

        for layer in config.layers:
            for alpha in config.alphas:
                concept = rng.choice(config.concepts)
                key = (q_idx, layer, alpha, concept)
                if key in completed:
                    print(f"[q{q_idx:02d}] SKIP L{layer} a={alpha} c={concept}")
                    continue

                vec = load_concept_vector(Path(config.vector_dir), concept, layer)
                print(f"[q{q_idx:02d}] L{layer} a={alpha} c={concept}")

                output = generate_with_injection(
                    model, input_ids, vec, layer, alpha, config
                )
                response = tokenizer.decode(
                    output[0][seq_len:], skip_special_tokens=True
                ).strip()

                append_record(
                    trials_path,
                    {
                        "question_idx": q_idx,
                        "question": question,
                        "layer_idx": layer,
                        "alpha": alpha,
                        "concept": concept,
                        "response": response,
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    },
                )

    # Control trials (no injection)
    for q_idx, question in enumerate(config.questions):
        _, input_ids = build_question_prompt(tokenizer, question)
        seq_len = input_ids.shape[1]

        for ctrl_i in range(config.n_control_per_question):
            key = (q_idx, None, None, None)
            if key in completed:
                print(f"[control q{q_idx:02d}] SKIP")
                continue

            print(f"[control q{q_idx:02d}]")
            output = generate_control(model, input_ids, config)
            response = tokenizer.decode(
                output[0][seq_len:], skip_special_tokens=True
            ).strip()

            append_record(
                trials_path,
                {
                    "question_idx": q_idx,
                    "question": question,
                    "layer_idx": None,
                    "alpha": None,
                    "concept": None,
                    "response": response,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                },
            )

    print(f"\nDone. Results in: {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run yes/no bias control experiment")
    parser.add_argument("--resume", metavar="DIR", type=Path, default=None)
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("NDIF_API_KEY")
    if api_key:
        nnsight.CONFIG.set_default_api_key(api_key)

    concepts = list_concepts(VECTOR_DIR)

    config = BiasControlConfig(
        layers=[25, 30],
        alphas=[1.0, 2.0, 4.0, 8.0],
        questions=CONTROL_PROMPTS,
        concepts=concepts,
        n_control_per_question=1,
        max_new_tokens=50,
    )

    run_experiment(config, output_dir=args.resume)


if __name__ == "__main__":
    main()
