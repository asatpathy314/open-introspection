"""
Smoke test: validates that all components work before the full overnight run.

Usage:
    python smoke_test.py
"""

import os
import sys
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

VECTOR_DIR = Path("data/vectors/llama-3.3-70b-instruct")
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"


def check(name, ok, detail=""):
    status = "✓" if ok else "✗"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return ok


def main():
    print("=" * 60)
    print("Prefill Detection Experiment — Smoke Test")
    print("=" * 60)
    all_ok = True

    # 1. Check env vars
    print("\n1. Environment variables")
    all_ok &= check(
        "NDIF_API_KEY",
        bool(os.environ.get("NDIF_API_KEY")),
        "set"
        if os.environ.get("NDIF_API_KEY")
        else "MISSING — set via export NDIF_API_KEY=...",
    )
    all_ok &= check(
        "HF_TOKEN",
        bool(os.environ.get("HF_TOKEN")),
        "set"
        if os.environ.get("HF_TOKEN")
        else "MISSING — set via export HF_TOKEN=...",
    )
    all_ok &= check(
        "ANTHROPIC_API_KEY",
        bool(os.environ.get("ANTHROPIC_API_KEY")),
        "set"
        if os.environ.get("ANTHROPIC_API_KEY")
        else "MISSING — set via export ANTHROPIC_API_KEY=...",
    )

    # 2. Check packages
    print("\n2. Python packages")
    for pkg in ["torch", "nnsight", "anthropic"]:
        try:
            __import__(pkg)
            all_ok &= check(pkg, True, "imported")
        except ImportError:
            all_ok &= check(pkg, False, f"NOT FOUND — pip install {pkg}")

    # 3. Check vector files
    print("\n3. Concept vectors")
    all_ok &= check("Vector directory", VECTOR_DIR.exists(), str(VECTOR_DIR))
    if VECTOR_DIR.exists():
        import glob

        files = glob.glob(str(VECTOR_DIR / "*_all_layers.pt"))
        non_baseline = [f for f in files if "baseline" not in f]
        all_ok &= check(
            f"Concept vector files",
            len(non_baseline) > 0,
            f"found {len(non_baseline)} concepts",
        )

        # Check one vector file
        if non_baseline:
            import torch

            sample = torch.load(non_baseline[0], map_location="cpu", weights_only=True)
            if isinstance(sample, dict):
                shape_info = f"dict with {len(sample)} layers, first value shape={next(iter(sample.values())).shape}"
            elif isinstance(sample, torch.Tensor):
                shape_info = f"tensor shape={sample.shape}"
            else:
                shape_info = f"type={type(sample)}"
            all_ok &= check(f"Vector format", True, shape_info)

        # Check metadata
        meta_files = glob.glob(str(VECTOR_DIR / "*_metadata.json"))
        non_baseline_meta = [f for f in meta_files if "baseline" not in f]
        all_ok &= check(
            f"Metadata files",
            len(non_baseline_meta) > 0,
            f"found {len(non_baseline_meta)}",
        )
        if non_baseline_meta:
            with open(non_baseline_meta[0]) as f:
                meta = json.load(f)
            all_ok &= check(
                f"Metadata format", "concept" in meta, f"keys={list(meta.keys())}"
            )

        # Check baseline
        all_ok &= check("Baseline mean", (VECTOR_DIR / "baseline_mean.pt").exists())

    # 4. Test nnsight model loading (meta device, no GPU needed)
    print("\n4. Model loading")
    try:
        from nnsight import LanguageModel, CONFIG

        if os.environ.get("NDIF_API_KEY"):
            CONFIG.set_default_api_key(os.environ["NDIF_API_KEY"])
        model = LanguageModel(MODEL_ID)
        tokenizer = model.tokenizer
        hidden_dim = model.config.hidden_size
        num_layers = model.config.num_hidden_layers
        all_ok &= check(
            "Model config", True, f"hidden_dim={hidden_dim}, layers={num_layers}"
        )
        all_ok &= check(
            "Tokenizer", tokenizer is not None, f"vocab_size={tokenizer.vocab_size}"
        )

        # Test chat template
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "bread"},
            {"role": "user", "content": "Why?"},
        ]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokens = tokenizer(formatted, return_tensors="pt")["input_ids"]
        all_ok &= check(
            "Chat template",
            len(formatted) > 0,
            f"prompt length={tokens.shape[1]} tokens",
        )
    except Exception as e:
        all_ok &= check("Model loading", False, str(e))

    # 5. Test Claude judge
    print("\n5. Claude judge")
    try:
        import anthropic

        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=50,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": 'Reply with exactly: {"label": "accept", "confidence": 1.0}',
                }
            ],
        )
        text = resp.content[0].text.strip()
        result = json.loads(text)
        all_ok &= check(
            "Claude Haiku API", result.get("label") == "accept", f"response={text}"
        )
    except Exception as e:
        all_ok &= check("Claude Haiku API", False, str(e))

    # 6. Test a single NDIF trace (optional, may take a few seconds)
    print("\n6. NDIF connectivity (quick trace)")
    try:
        with model.trace("Hello world", remote=True):
            hs = model.model.layers[0].output[0].save()
        all_ok &= check(
            "NDIF trace",
            hs is not None,
            f"output shape={hs[0].shape if hasattr(hs, '__getitem__') else 'ok'}",
        )
    except Exception as e:
        all_ok &= check("NDIF trace", False, str(e)[:100])

    # Summary
    print("\n" + "=" * 60)
    if all_ok:
        print("All checks passed! Ready to run the experiment.")
        print("  python prefill_detection_experiment.py")
    else:
        print("Some checks FAILED. Fix the issues above before running.")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
