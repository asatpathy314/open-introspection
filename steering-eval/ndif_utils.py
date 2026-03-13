"""NDIF/nnsight utilities — setup and retry only.

IMPORTANT: nnsight 0.6 does source-code analysis for trace contexts. All proxy
operations (layer output access, .save()) MUST be in the same file that is
executed as __main__. Cross-module function calls containing `with model.trace()`
will fail with WithBlockNotFoundError.

Shapes on NDIF:
  - model.model.layers[i].output[0] -> [seq_len, hidden_dim]  (no batch dim)
  - model.lm_head.output           -> [1, seq_len, vocab_size]
"""

import os
import time
import logging

import nnsight
from nnsight import CONFIG

from config import MODEL_ID, MAX_RETRIES, RETRY_DELAY

log = logging.getLogger(__name__)


def setup_ndif(model_id: str = MODEL_ID) -> nnsight.LanguageModel:
    """Initialize NDIF connection and return model handle."""
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.environ.get("NDIF_API_KEY")
    if not api_key:
        raise RuntimeError("NDIF_API_KEY not set")
    CONFIG.set_default_api_key(api_key)

    if not nnsight.is_model_running(model_id):
        raise RuntimeError(f"{model_id} is not online on NDIF")

    model = nnsight.LanguageModel(model_id)
    log.info("Model loaded: %s", model_id)
    return model


def retry(fn, *args, max_retries=MAX_RETRIES, delay=RETRY_DELAY, **kwargs):
    """Retry a callable with exponential backoff."""
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            log.warning("Attempt %d/%d failed: %s", attempt, max_retries, e)
            if attempt == max_retries:
                raise
            time.sleep(delay * attempt)
