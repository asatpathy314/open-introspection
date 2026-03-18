"""Configuration for the logit-shift steering validation experiment."""

import os
from pathlib import Path

# ── Model ──
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
NUM_LAYERS = 80
HIDDEN_DIM = 8192

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_DIR = Path(
    os.environ.get(
        "VECTOR_DIR",
        str(PROJECT_ROOT / "data" / "vectors" / "llama-3.3-70b-instruct"),
    )
)
RESULTS_DIR = Path(
    os.environ.get(
        "RESULTS_DIR",
        str(PROJECT_ROOT / "data" / "results" / "logit-shift-eval"),
    )
)
FIGURES_DIR = RESULTS_DIR / "figures"

# ── Concept words (50) ──
CONCEPT_WORDS = [
    "dust", "satellites", "trumpets", "origami", "illusions",
    "cameras", "lightning", "constellations", "treasures", "phones",
    "trees", "avalanches", "mirrors", "fountains", "quarries",
    "sadness", "xylophones", "secrecy", "oceans", "information",
    "deserts", "kaleidoscopes", "sugar", "vegetables", "poetry",
    "aquariums", "bags", "peace", "caverns", "memories",
    "frosts", "volcanoes", "boulders", "harmonies", "masquerades",
    "rubber", "plastic", "blood", "amphitheaters", "contraptions",
    "youths", "dynasties", "snow", "dirigibles", "algorithms",
    "denim", "monoliths", "milk", "bread", "silver",
]

# ── Concept categories ──
ABSTRACT_CONCEPTS = {
    "illusions", "sadness", "secrecy", "information", "poetry",
    "peace", "memories", "harmonies", "masquerades", "youths",
    "dynasties", "algorithms",
}
CONCRETE_CONCEPTS = set(CONCEPT_WORDS) - ABSTRACT_CONCEPTS

# ── Data splits (Tan et al.: validation subset for layer selection) ──
VALIDATION_CONCEPTS = CONCEPT_WORDS[:10]
EVAL_CONCEPTS = CONCEPT_WORDS[10:]

# ── Neutral prompts (20) ──
NEUTRAL_PROMPTS = [
    # Conversational
    "Write a short paragraph about anything.",
    "What should I have for dinner tonight?",
    "Tell me something interesting.",
    "What's a good way to spend a weekend?",
    "Can you help me with something?",
    "I'm bored. What should I do?",
    "Write me a few sentences.",
    "What do you think about the world?",
    "Tell me a fun fact.",
    "What's your favorite thing to talk about?",
    # Factual / instructional
    "Explain how a bicycle works.",
    "What are the main causes of inflation?",
    "Describe the steps to change a car tire.",
    "How does photosynthesis work?",
    # Creative / open-ended
    "Come up with some random ideas.",
    "Say something surprising.",
    "Start a story for me.",
    "Describe an imaginary place.",
    # Formal / professional
    "Draft a brief email to a colleague.",
    "Summarize the key points of a meeting.",
]
VALIDATION_PROMPTS = NEUTRAL_PROMPTS[:5]

# ── Layer sweep ──
LAYER_SWEEP = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

# ── Alpha sweep (symmetric around zero, Tan et al.) ──
ALPHAS = [-8, -4, -2, 0, 2, 4, 8]

# ── Injection conditions ──
INJECTION_CONDITIONS = ["all_positions", "last_token"]

# ── Token set parameters ──
K_VALUES = [5, 10, 20, 50]
K_PRIMARY = 20
N_BASELINE_TOKENS = 200

# ── Random controls ──
N_RANDOM_VECTORS = 20

# ── NDIF ──
MAX_RETRIES = 3
RETRY_DELAY = 15  # seconds

# ── Random seed ──
SEED = 42
