"""Shared configuration for the steering evaluation harness."""

import os
from pathlib import Path

# ── Model ──
MODEL_ID = os.environ.get("MODEL_ID", "meta-llama/Llama-3.1-70B-Instruct")
BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "meta-llama/Llama-3.1-70B")
NUM_LAYERS = 80
HIDDEN_DIM = 8192

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_DIR = Path(
    os.environ.get(
        "VECTOR_DIR",
        str(PROJECT_ROOT / "data" / "vectors" / "llama-3.1-70b-instruct"),
    )
)
RESULTS_DIR = Path(
    os.environ.get(
        "RESULTS_DIR",
        str(PROJECT_ROOT / "data" / "results" / "steering_eval"),
    )
)
FIGURES_DIR = RESULTS_DIR / "figures"

# ── Concept words (50) ──
CONCEPT_WORDS = [
    "dust", "satellites", "trumpets", "origami", "illusions", "cameras",
    "lightning", "constellations", "treasures", "phones", "trees", "avalanches",
    "mirrors", "fountains", "quarries", "sadness", "xylophones", "secrecy",
    "oceans", "information", "deserts", "kaleidoscopes", "sugar", "vegetables",
    "poetry", "aquariums", "bags", "peace", "caverns", "memories", "frosts",
    "volcanoes", "boulders", "harmonies", "masquerades", "rubber", "plastic",
    "blood", "amphitheaters", "contraptions", "youths", "dynasties", "snow",
    "dirigibles", "algorithms", "denim", "monoliths", "milk", "bread", "silver",
]

# ── Data split (Tan et al.: 40 eval, 10 validation for layer/strength selection) ──
VALIDATION_CONCEPTS = CONCEPT_WORDS[:10]  # first 10 for validation
EVAL_CONCEPTS = CONCEPT_WORDS[10:]  # remaining 40 for evaluation

# ── Layer sweep ──
LAYERS_TO_TEST = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 79]

# ── Alpha sweeps per normalization variant ──
ALPHA_SWEEP = {
    "raw": [0, 1, 2, 4, 8, 16],
    "unit": [0, 0.5, 1, 2, 4, 8],
    "norm_matched": [0, 0.1, 0.25, 0.5, 1.0, 2.0],
}

# ── NDIF / generation ──
MAX_RETRIES = 3
RETRY_DELAY = 15  # seconds
GENERATION_MAX_TOKENS = 100
GENERATION_TEMPERATURE = 1.0
GENERATION_TOP_P = 0.9

# ── Random seed ──
SEED = 42
