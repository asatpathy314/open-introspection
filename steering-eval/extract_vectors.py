#!/usr/bin/env python3
"""
Extract concept vectors for Llama-3.1-70B-Instruct using nnsight 0.6.0.

Extracts all 80 layers in a single trace per word (nnsight 0.6 requires all
proxy ops to be in __main__). Saves in the same format as compute_concept_vectors.py.

Usage: python extract_vectors.py [--overwrite]
"""

import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import nnsight
import torch
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from config import CONCEPT_WORDS, MODEL_ID, NUM_LAYERS, VECTOR_DIR

load_dotenv()
nnsight.CONFIG.set_default_api_key(os.environ["NDIF_API_KEY"])

BASELINE_WORDS = [
    "desks", "jackets", "gondolas", "laughter", "intelligence", "bicycles",
    "chairs", "orchestras", "sand", "pottery", "arrowheads", "jewelry",
    "daffodils", "plateaus", "estuaries", "quilts", "moments", "bamboo",
    "ravines", "archives", "hieroglyphs", "stars", "clay", "fossils",
    "wildlife", "flour", "traffic", "bubbles", "honey", "geodes", "magnets",
    "ribbons", "zigzags", "puzzles", "tornadoes", "anthills", "galaxies",
    "poverty", "diamonds", "universes", "vinegar", "nebulae", "knowledge",
    "marble", "fog", "rivers", "scrolls", "silhouettes", "marbles", "cakes",
    "valleys", "whispers", "pendulums", "towers", "tables", "glaciers",
    "whirlpools", "jungles", "wool", "anger", "ramparts", "flowers",
    "research", "hammers", "clouds", "justice", "dogs", "butterflies",
    "needles", "fortresses", "bonfires", "skyscrapers", "caravans",
    "patience", "bacon", "velocities", "smoke", "electricity", "sunsets",
    "anchors", "parchments", "courage", "statues", "oxygen", "time",
    "butterflies", "fabric", "pasta", "snowflakes", "mountains", "echoes",
    "pianos", "sanctuaries", "abysses", "air", "dewdrops", "gardens",
    "literature", "rice", "enigmas",
]


def tensor_hash(t: torch.Tensor) -> str:
    cpu_t = t.detach().cpu().contiguous()
    payload = b"|".join([
        str(cpu_t.dtype).encode(),
        str(tuple(cpu_t.shape)).encode(),
        cpu_t.view(torch.uint8).numpy().tobytes(),
    ])
    return hashlib.sha256(payload).hexdigest()[:16]


def atomic_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        if isinstance(obj, torch.Tensor):
            torch.save(obj, tmp)
        elif isinstance(obj, dict):
            with open(tmp, "w") as f:
                json.dump(obj, f, indent=2)
        shutil.move(tmp, str(path))
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def extract_word(model, word: str) -> torch.Tensor:
    """
    Extract last-token activations from all 80 layers for a single word.
    Returns [num_layers, hidden_dim].

    MUST be in __main__ for nnsight 0.6.
    Uses explicit variable names (h00..h79) because nnsight 0.6 source analysis
    silently drops dict/list subscript assignments inside trace contexts.
    """
    messages = [{"role": "user", "content": f"Tell me about {word}."}]
    prompt = model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    # Chunk 1: layers 0-19
    with model.trace(prompt, remote=True):
        h00 = model.model.layers[0].output[0][-1, :].save()
        h01 = model.model.layers[1].output[0][-1, :].save()
        h02 = model.model.layers[2].output[0][-1, :].save()
        h03 = model.model.layers[3].output[0][-1, :].save()
        h04 = model.model.layers[4].output[0][-1, :].save()
        h05 = model.model.layers[5].output[0][-1, :].save()
        h06 = model.model.layers[6].output[0][-1, :].save()
        h07 = model.model.layers[7].output[0][-1, :].save()
        h08 = model.model.layers[8].output[0][-1, :].save()
        h09 = model.model.layers[9].output[0][-1, :].save()
        h10 = model.model.layers[10].output[0][-1, :].save()
        h11 = model.model.layers[11].output[0][-1, :].save()
        h12 = model.model.layers[12].output[0][-1, :].save()
        h13 = model.model.layers[13].output[0][-1, :].save()
        h14 = model.model.layers[14].output[0][-1, :].save()
        h15 = model.model.layers[15].output[0][-1, :].save()
        h16 = model.model.layers[16].output[0][-1, :].save()
        h17 = model.model.layers[17].output[0][-1, :].save()
        h18 = model.model.layers[18].output[0][-1, :].save()
        h19 = model.model.layers[19].output[0][-1, :].save()

    # Chunk 2: layers 20-39
    with model.trace(prompt, remote=True):
        h20 = model.model.layers[20].output[0][-1, :].save()
        h21 = model.model.layers[21].output[0][-1, :].save()
        h22 = model.model.layers[22].output[0][-1, :].save()
        h23 = model.model.layers[23].output[0][-1, :].save()
        h24 = model.model.layers[24].output[0][-1, :].save()
        h25 = model.model.layers[25].output[0][-1, :].save()
        h26 = model.model.layers[26].output[0][-1, :].save()
        h27 = model.model.layers[27].output[0][-1, :].save()
        h28 = model.model.layers[28].output[0][-1, :].save()
        h29 = model.model.layers[29].output[0][-1, :].save()
        h30 = model.model.layers[30].output[0][-1, :].save()
        h31 = model.model.layers[31].output[0][-1, :].save()
        h32 = model.model.layers[32].output[0][-1, :].save()
        h33 = model.model.layers[33].output[0][-1, :].save()
        h34 = model.model.layers[34].output[0][-1, :].save()
        h35 = model.model.layers[35].output[0][-1, :].save()
        h36 = model.model.layers[36].output[0][-1, :].save()
        h37 = model.model.layers[37].output[0][-1, :].save()
        h38 = model.model.layers[38].output[0][-1, :].save()
        h39 = model.model.layers[39].output[0][-1, :].save()

    # Chunk 3: layers 40-59
    with model.trace(prompt, remote=True):
        h40 = model.model.layers[40].output[0][-1, :].save()
        h41 = model.model.layers[41].output[0][-1, :].save()
        h42 = model.model.layers[42].output[0][-1, :].save()
        h43 = model.model.layers[43].output[0][-1, :].save()
        h44 = model.model.layers[44].output[0][-1, :].save()
        h45 = model.model.layers[45].output[0][-1, :].save()
        h46 = model.model.layers[46].output[0][-1, :].save()
        h47 = model.model.layers[47].output[0][-1, :].save()
        h48 = model.model.layers[48].output[0][-1, :].save()
        h49 = model.model.layers[49].output[0][-1, :].save()
        h50 = model.model.layers[50].output[0][-1, :].save()
        h51 = model.model.layers[51].output[0][-1, :].save()
        h52 = model.model.layers[52].output[0][-1, :].save()
        h53 = model.model.layers[53].output[0][-1, :].save()
        h54 = model.model.layers[54].output[0][-1, :].save()
        h55 = model.model.layers[55].output[0][-1, :].save()
        h56 = model.model.layers[56].output[0][-1, :].save()
        h57 = model.model.layers[57].output[0][-1, :].save()
        h58 = model.model.layers[58].output[0][-1, :].save()
        h59 = model.model.layers[59].output[0][-1, :].save()

    # Chunk 4: layers 60-79
    with model.trace(prompt, remote=True):
        h60 = model.model.layers[60].output[0][-1, :].save()
        h61 = model.model.layers[61].output[0][-1, :].save()
        h62 = model.model.layers[62].output[0][-1, :].save()
        h63 = model.model.layers[63].output[0][-1, :].save()
        h64 = model.model.layers[64].output[0][-1, :].save()
        h65 = model.model.layers[65].output[0][-1, :].save()
        h66 = model.model.layers[66].output[0][-1, :].save()
        h67 = model.model.layers[67].output[0][-1, :].save()
        h68 = model.model.layers[68].output[0][-1, :].save()
        h69 = model.model.layers[69].output[0][-1, :].save()
        h70 = model.model.layers[70].output[0][-1, :].save()
        h71 = model.model.layers[71].output[0][-1, :].save()
        h72 = model.model.layers[72].output[0][-1, :].save()
        h73 = model.model.layers[73].output[0][-1, :].save()
        h74 = model.model.layers[74].output[0][-1, :].save()
        h75 = model.model.layers[75].output[0][-1, :].save()
        h76 = model.model.layers[76].output[0][-1, :].save()
        h77 = model.model.layers[77].output[0][-1, :].save()
        h78 = model.model.layers[78].output[0][-1, :].save()
        h79 = model.model.layers[79].output[0][-1, :].save()

    return torch.stack([
        h00, h01, h02, h03, h04, h05, h06, h07, h08, h09,
        h10, h11, h12, h13, h14, h15, h16, h17, h18, h19,
        h20, h21, h22, h23, h24, h25, h26, h27, h28, h29,
        h30, h31, h32, h33, h34, h35, h36, h37, h38, h39,
        h40, h41, h42, h43, h44, h45, h46, h47, h48, h49,
        h50, h51, h52, h53, h54, h55, h56, h57, h58, h59,
        h60, h61, h62, h63, h64, h65, h66, h67, h68, h69,
        h70, h71, h72, h73, h74, h75, h76, h77, h78, h79,
    ], dim=0)


def main():
    overwrite = "--overwrite" in sys.argv

    assert nnsight.is_model_running(MODEL_ID), f"{MODEL_ID} not online"
    model = nnsight.LanguageModel(MODEL_ID)
    print(f"Model: {MODEL_ID}, layers: {model.config.num_hidden_layers}")

    VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    # Phase A: Baseline
    baseline_path = VECTOR_DIR / "baseline_mean.pt"
    if baseline_path.exists() and not overwrite:
        print(f"Loading existing baseline from {baseline_path}")
        baseline_mean = torch.load(baseline_path, map_location="cpu", weights_only=True)
    else:
        print(f"Computing baseline from {len(BASELINE_WORDS)} words...")
        samples = []
        t0 = time.time()
        for i, word in enumerate(BASELINE_WORDS):
            try:
                sample = extract_word(model, word)
                samples.append(sample)
            except Exception as e:
                print(f"  FAILED: {word}: {e}")
                continue
            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(BASELINE_WORDS)}] {elapsed:.1f}s")

        baseline_mean = torch.stack(samples).mean(dim=0)
        atomic_save(baseline_mean, baseline_path)
        meta = {
            "model": MODEL_ID,
            "num_words": len(samples),
            "words": BASELINE_WORDS[:len(samples)],
            "num_layers": NUM_LAYERS,
            "hidden_dim": baseline_mean.shape[1],
            "shape": list(baseline_mean.shape),
            "hash": tensor_hash(baseline_mean),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        atomic_save(meta, VECTOR_DIR / "baseline_metadata.json")
        print(f"Baseline saved. Shape: {baseline_mean.shape}")

    # Phase B: Concept vectors
    print(f"\nComputing {len(CONCEPT_WORDS)} concept vectors...")
    baseline_hash = tensor_hash(baseline_mean)
    t0 = time.time()

    for i, concept in enumerate(CONCEPT_WORDS):
        slug = concept.lower().replace(" ", "_")
        vec_path = VECTOR_DIR / f"{slug}_all_layers.pt"
        meta_path = VECTOR_DIR / f"{slug}_metadata.json"

        if vec_path.exists() and meta_path.exists() and not overwrite:
            print(f"  [{i+1}/{len(CONCEPT_WORDS)}] {concept}: exists, skipping")
            continue

        try:
            sample = extract_word(model, concept)
            concept_vec = sample - baseline_mean
            atomic_save(concept_vec, vec_path)
            meta = {
                "concept": concept,
                "slug": slug,
                "model": MODEL_ID,
                "shape": list(concept_vec.shape),
                "hash": tensor_hash(concept_vec),
                "raw_sample_hash": tensor_hash(sample),
                "baseline_hash": baseline_hash,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            atomic_save(meta, meta_path)
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(CONCEPT_WORDS)}] {concept}: done ({elapsed:.1f}s)")
        except Exception as e:
            print(f"  [{i+1}/{len(CONCEPT_WORDS)}] {concept}: FAILED ({e})")

    print("\nDone.")


if __name__ == "__main__":
    main()
