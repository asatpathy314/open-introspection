"""NDIF API key configuration for the injected-thoughts experiment."""

import os

import nnsight
from dotenv import load_dotenv


def configure_ndif_api_key() -> str:
    """Load NDIF API key from environment / .env and register it with nnsight.

    Assumptions:
        - NDIF_API_KEY is present in the environment or in a .env file at or above
          the working directory.

    Returns:
        str: The API key that was registered.

    Raises:
        RuntimeError: If NDIF_API_KEY is not found.
    """
    load_dotenv()
    api_key = os.environ.get("NDIF_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NDIF_API_KEY was not found. Add it to your environment or .env file."
        )
    nnsight.CONFIG.set_default_api_key(api_key)  # type: ignore
    return api_key
