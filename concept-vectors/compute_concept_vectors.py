#!/usr/bin/env python3
"""
compute_concept_vectors.py

Extracts concept vectors following Lindsey et al. ("Emergent Introspective Awareness")
using nnsight + NDIF remote execution on Llama-3.3-70B-Instruct.

Two-phase execution:
  Phase A: Compute baseline vectors in parallel using thread-local model instances.
  Phase B: Compute all concept vectors in parallel using thread-local model instances.

Filename convention for saved vectors:
  data/vectors/llama-3.3-70b-instruct/<concept_slug>_all_layers.pt
  data/vectors/llama-3.3-70b-instruct/<concept_slug>_metadata.json
  data/vectors/llama-3.3-70b-instruct/baseline_mean.pt
  data/vectors/llama-3.3-70b-instruct/baseline_metadata.json

Where <concept_slug> is the lowercase, stripped concept name with spaces replaced by underscores.
"""

import os
import re
import json
import time
import hashlib
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import torch
import nnsight
from dotenv import load_dotenv

# ============================================================================
# Global Configuration
# ============================================================================

MODEL = os.environ.get("MODEL", "meta-llama/Llama-3.3-70B-Instruct")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "data/vectors/llama-3.3-70b-instruct"))
MAX_IN_FLIGHT = int(os.environ.get("MAX_IN_FLIGHT", "4"))
OVERWRITE = os.environ.get("OVERWRITE", "").lower() in ("1", "true", "yes")
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "2.0"))
POLL_TIMEOUT = float(os.environ.get("POLL_TIMEOUT", "600.0"))

# Verification thresholds
ATOL = float(os.environ.get("ATOL", "1e-3"))
COS_MIN = float(os.environ.get("COS_MIN", "0.9999"))
VERIFY_N = int(os.environ.get("VERIFY_N", "5"))

# Lindsey et al. baseline words (100 words, as in the ground-truth code)
BASELINE_WORDS_STR = (
    "Desks, Jackets, Gondolas, Laughter, Intelligence, Bicycles, Chairs, "
    "Orchestras, Sand, Pottery, Arrowheads, Jewelry, Daffodils, Plateaus, "
    "Estuaries, Quilts, Moments, Bamboo, Ravines, Archives, Hieroglyphs, "
    "Stars, Clay, Fossils, Wildlife, Flour, Traffic, Bubbles, Honey, Geodes, "
    "Magnets, Ribbons, Zigzags, Puzzles, Tornadoes, Anthills, Galaxies, "
    "Poverty, Diamonds, Universes, Vinegar, Nebulae, Knowledge, Marble, Fog, "
    "Rivers, Scrolls, Silhouettes, Marbles, Cakes, Valleys, Whispers, "
    "Pendulums, Towers, Tables, Glaciers, Whirlpools, Jungles, Wool, Anger, "
    "Ramparts, Flowers, Research, Hammers, Clouds, Justice, Dogs, Butterflies, "
    "Needles, Fortresses, Bonfires, Skyscrapers, Caravans, Patience, Bacon, "
    "Velocities, Smoke, Electricity, Sunsets, Anchors, Parchments, Courage, "
    "Statues, Oxygen, Time, Butterflies, Fabric, Pasta, Snowflakes, Mountains, "
    "Echoes, Pianos, Sanctuaries, Abysses, Air, Dewdrops, Gardens, Literature, "
    "Rice, Enigmas"
)
BASELINE_WORDS = [w.strip().lower() for w in BASELINE_WORDS_STR.split(",") if w.strip()]

# Concepts to compute vectors for. By default, use a sample set.
# The ground truth code demonstrates with "Dust"; extend as needed.
CONCEPTS = (
    "Dust, Satellites, Trumpets, Origami, Illusions, Cameras, Lightning, "
    "Constellations, Treasures, Phones, Trees, Avalanches, Mirrors, Fountains, "
    "Quarries, Sadness, Xylophones, Secrecy, Oceans, Information, Deserts, "
    "Kaleidoscopes, Sugar, Vegetables, Poetry, Aquariums, Bags, Peace, Caverns, "
    "Memories, Frosts, Volcanoes, Boulders, Harmonies, Masquerades, Rubber, Plastic, "
    "Blood, Amphitheaters, Contraptions, Youths, Dynasties, Snow, Dirigibles, Algorithms, "
    "Denim, Monoliths, Milk, Bread, Silver"
)
CONCEPT_LIST = [w.strip().lower() for w in CONCEPTS.split(",") if w.strip()]
_THREAD_LOCAL = threading.local()

# ============================================================================
# Helpers
# ============================================================================

def concept_slug(concept: str) -> str:
    """Deterministic filename-safe slug for a concept."""
    return re.sub(r"[^a-z0-9]+", "_", concept.lower()).strip("_")


def configure_ndif_api_key() -> str:
    """
    Load NDIF API key from environment/.env and register it with nnsight.
    Returns the key for confirmation/logging if needed.
    """
    load_dotenv()
    api_key = os.environ.get("NDIF_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NDIF_API_KEY was not found. Add it to your environment or .env file."
        )
    nnsight.CONFIG.set_default_api_key(api_key)
    return api_key


def get_thread_model():
    """Return one LanguageModel per worker thread (remote execution, no weight load)."""
    if not hasattr(_THREAD_LOCAL, "model"):
        _THREAD_LOCAL.model = nnsight.LanguageModel(MODEL)
    return _THREAD_LOCAL.model


def make_prompt(model, word: str) -> str:
    """Build the chat-template prompt for Llama-3 models."""
    messages = [{"role": "user", "content": f"Tell me about {word}."}]
    return model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def tensor_hash(t: torch.Tensor) -> str:
    """SHA-256 of a tensor's bytes for reproducibility checks."""
    cpu_tensor = t.detach().cpu().contiguous()
    payload = b"|".join(
        [
            str(cpu_tensor.dtype).encode("utf-8"),
            str(tuple(cpu_tensor.shape)).encode("utf-8"),
            cpu_tensor.view(torch.uint8).numpy().tobytes(),
        ]
    )
    return hashlib.sha256(payload).hexdigest()[:16]


def atomic_save(obj, path: Path):
    """Save a torch tensor or JSON dict atomically (write to tmp then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        if isinstance(obj, torch.Tensor):
            torch.save(obj, tmp)
        elif isinstance(obj, dict):
            with open(tmp, "w") as f:
                json.dump(obj, f, indent=2)
        else:
            raise TypeError(f"Cannot save object of type {type(obj)}")
        shutil.move(tmp, str(path))
    except:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def extract_activations_from_cache(cache, num_layers: int) -> torch.Tensor:
    """
    Extract last-token activations from all layers:
        cache[f'model.model.layers.{i}']["output"][0, -1, :]
    Returns: tensor of shape [num_layers, hidden_dim]
    """
    return torch.stack(
        [
            cache[f"model.model.layers.{i}"]["output"][0, -1, :]
            for i in range(num_layers)
        ],
        dim=0,
    )


def poll_backend(backend, timeout: float = POLL_TIMEOUT, interval: float = POLL_INTERVAL):
    """Poll a non-blocking backend until the result is ready or timeout."""
    start = time.time()
    while True:
        result = backend()
        if result is not None:
            return result
        if time.time() - start > timeout:
            raise TimeoutError(
                f"Backend poll timed out after {timeout}s. "
                f"Status: {backend.job_status}"
            )
        time.sleep(interval)


# ============================================================================
# Phase A: Baseline (parallel, thread-local models)
# ============================================================================

def compute_baseline_vectors(
    model,
    words: List[str] = BASELINE_WORDS,
    max_in_flight: int = MAX_IN_FLIGHT,
) -> torch.Tensor:
    """
    Compute the mean baseline last-token activation vector using the given
    words, following the exact procedure from the ground-truth code.

    Returns: baseline_mean tensor of shape [num_layers, hidden_dim]
    Saves: baseline_mean.pt and baseline_metadata.json to OUTPUT_DIR.
    """
    num_layers = model.config.num_hidden_layers
    baseline_samples = []

    print(
        f"[Phase A] Computing baseline from {len(words)} words "
        f"(parallel, max_in_flight={max_in_flight})..."
    )
    t0 = time.time()
    completed_count = [0]
    lock = threading.Lock()

    def process_baseline_word(word: str) -> torch.Tensor:
        worker_model = get_thread_model()
        return _trace_word_blocking(worker_model, word, num_layers)

    with ThreadPoolExecutor(max_workers=max_in_flight) as executor:
        futures = {executor.submit(process_baseline_word, w): w for w in words}
        for future in as_completed(futures):
            word = futures[future]
            try:
                baseline_samples.append(future.result())
            except Exception as exc:
                raise RuntimeError(
                    f"Baseline trace failed for word={word!r}: {type(exc).__name__}: {exc}"
                ) from exc

            with lock:
                completed_count[0] += 1
                n_done = completed_count[0]
            if n_done % 10 == 0 or n_done == len(words):
                elapsed = time.time() - t0
                rate = n_done / elapsed
                print(f"  [{n_done}/{len(words)}] {rate:.1f} words/s, elapsed {elapsed:.1f}s")

    baseline_stack = torch.stack(baseline_samples, dim=0)  # [N, num_layers, hidden]
    baseline_mean = baseline_stack.mean(dim=0)              # [num_layers, hidden]

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_save(baseline_mean, OUTPUT_DIR / "baseline_mean.pt")
    metadata = {
        "model": MODEL,
        "num_words": len(words),
        "words": words,
        "num_layers": num_layers,
        "hidden_dim": baseline_mean.shape[1],
        "shape": list(baseline_mean.shape),
        "hash": tensor_hash(baseline_mean),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    atomic_save(metadata, OUTPUT_DIR / "baseline_metadata.json")

    elapsed = time.time() - t0
    print(f"[Phase A] Baseline computed in {elapsed:.1f}s. Shape: {baseline_mean.shape}")
    print(f"          Saved to {OUTPUT_DIR / 'baseline_mean.pt'}")
    return baseline_mean


def load_baseline(path: Optional[Path] = None) -> torch.Tensor:
    """Load a previously saved baseline_mean tensor."""
    path = path or (OUTPUT_DIR / "baseline_mean.pt")
    assert path.exists(), f"Baseline not found at {path}"
    return torch.load(path, weights_only=True)


# ============================================================================
# Phase B: Async / parallel concept vector computation
# ============================================================================

def _trace_word_blocking(model, word: str, num_layers: int) -> torch.Tensor:
    """Run a single blocking trace for a word. Returns [num_layers, hidden]."""
    prompt = make_prompt(model, word)
    with model.trace(prompt, remote=True) as tracer:
        cache = tracer.cache(
            modules=[layer for layer in model.model.layers]
        ).save()
    return extract_activations_from_cache(cache, num_layers)



def compute_concept_vector(
    sample: torch.Tensor,
    baseline_mean: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the concept vector: sample - baseline_mean, per layer.
    Both are [num_layers, hidden_dim].
    Returns [num_layers, hidden_dim].
    """
    return sample - baseline_mean


def save_concept_vector(
    concept: str,
    concept_vector: torch.Tensor,
    raw_sample: torch.Tensor,
    baseline_hash: str,
):
    """Save a concept vector and its metadata atomically."""
    slug = concept_slug(concept)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save the full [num_layers, hidden] concept vector
    vec_path = OUTPUT_DIR / f"{slug}_all_layers.pt"
    atomic_save(concept_vector, vec_path)

    # Save metadata
    meta = {
        "concept": concept,
        "slug": slug,
        "model": MODEL,
        "shape": list(concept_vector.shape),
        "hash": tensor_hash(concept_vector),
        "raw_sample_hash": tensor_hash(raw_sample),
        "baseline_hash": baseline_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    atomic_save(meta, OUTPUT_DIR / f"{slug}_metadata.json")
    return vec_path


def concept_already_on_disk(concept: str) -> bool:
    """Check if a concept vector already exists on disk."""
    slug = concept_slug(concept)
    vec_path = OUTPUT_DIR / f"{slug}_all_layers.pt"
    meta_path = OUTPUT_DIR / f"{slug}_metadata.json"
    return vec_path.exists() and meta_path.exists()


def compute_all_vectors_async(
    model,
    concepts: List[str],
    baseline_mean: torch.Tensor,
    max_in_flight: int = MAX_IN_FLIGHT,
):
    """
    Phase B: Compute all concept vectors in parallel using thread-local
    LanguageModel instances to avoid trace-state races.
    """
    num_layers = model.config.num_hidden_layers
    baseline_hash = tensor_hash(baseline_mean)

    # Filter concepts
    if OVERWRITE:
        to_compute = concepts
    else:
        to_compute = [c for c in concepts if not concept_already_on_disk(c)]
        skipped = len(concepts) - len(to_compute)
        if skipped:
            print(f"[Phase B] Skipping {skipped} concepts already on disk (OVERWRITE=False)")

    if not to_compute:
        print("[Phase B] Nothing to compute.")
        return []

    print(f"[Phase B] Computing {len(to_compute)} concept vectors "
          f"(max_in_flight={max_in_flight})...")
    t0 = time.time()

    failures = []
    completed_count = [0]
    lock = threading.Lock()

    def process_concept(concept: str) -> Optional[str]:
        try:
            worker_model = get_thread_model()
            sample = _trace_word_blocking(worker_model, concept, num_layers)
            cvec = compute_concept_vector(sample, baseline_mean)
            save_concept_vector(concept, cvec, sample, baseline_hash)

            with lock:
                completed_count[0] += 1
                n_done = completed_count[0]
            elapsed = time.time() - t0
            print(
                f"  [{n_done}/{len(to_compute)}] '{concept}' done "
                f"({elapsed:.1f}s elapsed)"
            )
            return None

        except Exception as exc:
            err = f"'{concept}': {type(exc).__name__}: {exc}"
            print(f"  [ERROR] {err}")
            return err

    with ThreadPoolExecutor(max_workers=max_in_flight) as executor:
        futures = {executor.submit(process_concept, c): c for c in to_compute}
        for future in as_completed(futures):
            err = future.result()
            if err is not None:
                failures.append(err)

    elapsed = time.time() - t0
    print(f"[Phase B] Completed {completed_count[0]}/{len(to_compute)} in {elapsed:.1f}s")
    if failures:
        print(f"[Phase B] {len(failures)} failures:")
        for f in failures:
            print(f"  - {f}")

    return failures


# ============================================================================
# Verification
# ============================================================================

def verify_against_baseline(
    model,
    baseline_mean: torch.Tensor,
    n: int = VERIFY_N,
    atol: float = ATOL,
    cos_min: float = COS_MIN,
):
    """
    Verify that the async pipeline produces identical results to the blocking
    baseline for a small set of concepts.
    """
    verify_concepts = CONCEPT_LIST[:n]
    num_layers = model.config.num_hidden_layers
    baseline_hash = tensor_hash(baseline_mean)

    print(f"\n[Verify] Checking {len(verify_concepts)} concepts...")

    all_pass = True

    for concept in verify_concepts:
        # Method 1: Blocking (ground-truth equivalent)
        sample_blocking = _trace_word_blocking(model, concept, num_layers)
        vec_blocking = compute_concept_vector(sample_blocking, baseline_mean)

        # Method 2: Load from disk (computed by Phase B async)
        slug = concept_slug(concept)
        disk_path = OUTPUT_DIR / f"{slug}_all_layers.pt"

        if not disk_path.exists():
            print(f"  '{concept}': SKIP (not on disk)")
            continue

        vec_async = torch.load(disk_path, weights_only=True)

        # Check absolute difference
        max_diff = (vec_blocking - vec_async).abs().max().item()

        # Check cosine similarity (flatten to 1D for cos sim)
        cos_sim = torch.nn.functional.cosine_similarity(
            vec_blocking.flatten().unsqueeze(0),
            vec_async.flatten().unsqueeze(0),
        ).item()

        passed = max_diff <= atol and cos_sim >= cos_min
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False

        print(f"  '{concept}': {status} "
              f"(max_diff={max_diff:.2e}, cos_sim={cos_sim:.8f})")

    if all_pass:
        print("[Verify] All checks passed!")
    else:
        print("[Verify] Some checks FAILED — investigate differences.")

    return all_pass


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Concept Vector Extraction (Lindsey et al.)")
    print(f"  Model: {MODEL}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Baseline words: {len(BASELINE_WORDS)}")
    print(f"  Concepts: {len(CONCEPT_LIST)}")
    print(f"  Max in-flight: {MAX_IN_FLIGHT}")
    print(f"  Overwrite: {OVERWRITE}")
    print("=" * 70)

    configure_ndif_api_key()

    # Check NDIF is available
    assert nnsight.is_model_running(MODEL), f"{MODEL} is not online on NDIF."
    print(f"Model {MODEL} is online.")

    # Load model (lazy — no weights downloaded locally)
    model = nnsight.LanguageModel(MODEL)
    print(f"Model loaded. Layers: {model.config.num_hidden_layers}")

    # --- Phase A: Baseline ---
    baseline_path = OUTPUT_DIR / "baseline_mean.pt"
    if baseline_path.exists() and not OVERWRITE:
        print(f"\n[Phase A] Loading existing baseline from {baseline_path}")
        baseline_mean = load_baseline(baseline_path)
        print(f"          Shape: {baseline_mean.shape}, hash: {tensor_hash(baseline_mean)}")
    else:
        print()
        baseline_mean = compute_baseline_vectors(model)

    # --- Phase B: All concept vectors (async) ---
    print()
    failures = compute_all_vectors_async(model, CONCEPT_LIST, baseline_mean) or []

    # --- Verification ---
    verify_against_baseline(model, baseline_mean)

    print("\nDone.")
    if failures:
        print(f"WARNING: {len(failures)} concept(s) failed. See above.")


if __name__ == "__main__":
    main()
