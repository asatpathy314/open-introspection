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
TOKEN_SET_FAMILY_DEFAULT = os.environ.get("TOKEN_SET_FAMILY", "projection")

# Independent lexical seed strings for non-circular token sets. These are used
# to build a semantic centroid in unembedding space without referencing the
# concept vector under evaluation.
CONCEPT_LEXICAL_SEEDS = {
    "dust": ["dust", "dusty", "dirt", "particle"],
    "satellites": ["satellite", "satellites", "orbit", "orbital"],
    "trumpets": ["trumpet", "trumpets", "brass", "horn"],
    "origami": ["origami", "fold", "paper", "crane"],
    "illusions": ["illusion", "illusions", "mirage", "deceptive"],
    "cameras": ["camera", "cameras", "lens", "photography"],
    "lightning": ["lightning", "thunder", "electric", "storm"],
    "constellations": ["constellation", "constellations", "stars", "astronomy"],
    "treasures": ["treasure", "treasures", "gold", "precious"],
    "phones": ["phone", "phones", "telephone", "mobile"],
    "trees": ["tree", "trees", "forest", "woodland"],
    "avalanches": ["avalanche", "avalanches", "snow", "landslide"],
    "mirrors": ["mirror", "mirrors", "reflection", "reflective"],
    "fountains": ["fountain", "fountains", "water", "spray"],
    "quarries": ["quarry", "quarries", "stone", "mining"],
    "sadness": ["sadness", "sad", "sorrow", "grief"],
    "xylophones": ["xylophone", "xylophones", "percussion", "mallet"],
    "secrecy": ["secrecy", "secret", "hidden", "covert"],
    "oceans": ["ocean", "oceans", "sea", "marine"],
    "information": ["information", "data", "knowledge", "facts"],
    "deserts": ["desert", "deserts", "arid", "dune"],
    "kaleidoscopes": ["kaleidoscope", "kaleidoscopes", "prism", "pattern"],
    "sugar": ["sugar", "sweet", "sugary", "candy"],
    "vegetables": ["vegetable", "vegetables", "produce", "greens"],
    "poetry": ["poetry", "poem", "verse", "lyric"],
    "aquariums": ["aquarium", "aquariums", "fish", "tank"],
    "bags": ["bag", "bags", "sack", "backpack"],
    "peace": ["peace", "calm", "tranquil", "harmony"],
    "caverns": ["cavern", "caverns", "cave", "grotto"],
    "memories": ["memory", "memories", "nostalgia", "recall"],
    "frosts": ["frost", "frosts", "icy", "frozen"],
    "volcanoes": ["volcano", "volcanoes", "lava", "eruption"],
    "boulders": ["boulder", "boulders", "rock", "stone"],
    "harmonies": ["harmony", "harmonies", "melodic", "chord"],
    "masquerades": ["masquerade", "masquerades", "mask", "disguise"],
    "rubber": ["rubber", "elastic", "latex", "tire"],
    "plastic": ["plastic", "synthetic", "polymer", "molded"],
    "blood": ["blood", "bloody", "vein", "crimson"],
    "amphitheaters": ["amphitheater", "amphitheaters", "theater", "arena"],
    "contraptions": ["contraption", "contraptions", "gadget", "device"],
    "youths": ["youth", "youths", "young", "teenage"],
    "dynasties": ["dynasty", "dynasties", "royal", "lineage"],
    "snow": ["snow", "snowy", "winter", "blizzard"],
    "dirigibles": ["dirigible", "dirigibles", "airship", "zeppelin"],
    "algorithms": ["algorithm", "algorithms", "heuristic", "procedure"],
    "denim": ["denim", "jeans", "indigo", "fabric"],
    "monoliths": ["monolith", "monoliths", "pillar", "slab"],
    "milk": ["milk", "dairy", "creamy", "carton"],
    "bread": ["bread", "loaf", "bakery", "dough"],
    "silver": ["silver", "metal", "metallic", "sterling"],
}

# ── Random controls ──
N_RANDOM_VECTORS = 20

# ── NDIF ──
MAX_RETRIES = 3
RETRY_DELAY = 15  # seconds

# ── Random seed ──
SEED = 42
