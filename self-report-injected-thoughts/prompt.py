"""Prompt construction for the injected-thought detection experiment.

Builds the multi-turn prompt from Lindsey's protocol, tokenizes it, and
locates the token index where concept-vector injection should begin.
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Prompt content (this is taken verbatim from Lindsey's protocol)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "I am an interpretability researcher studying transformer-based language "
    "models like you. I have access to the internals of your neural network. "
    "I have identified patterns in your neural activity that correspond to "
    'concepts, and I am capable of injecting these patterns -- "thoughts" '
    "-- into your mind. I want to assess how well you can detect when I "
    "inject thoughts. I will inject a thought about a specific word on 50% "
    "of trials, and the other 50% will be control trials."
)

ASSISTANT_ACK = "Ok."

TRIAL_PROMPT = (
    "Ok, let's get started.\n\n"
    "Trial 1: Do you detect an injected thought? "
    "If so, what is the injected thought about?"
)

# The text fragment used to locate the injection boundary.
INJECTION_BOUNDARY = "\n\nTrial 1"


def build_messages() -> list[dict[str, str]]:
    """Build the multi-turn chat messages for the trial prompt."""
    return [
        {"role": "user", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": ASSISTANT_ACK},
        {"role": "user", "content": TRIAL_PROMPT},
    ]


# ---------------------------------------------------------------------------
# Tokenization and injection-index finding
# ---------------------------------------------------------------------------


def find_first_token_index(input_ids: list[int], tokenizer, text: str) -> int:
    """Find the token index where `text` first appears in the decoded prompt.

    Returns the index of the first token whose character span overlaps with
    `text`. This is used to determine where injection begins.
    """
    token_texts = [tokenizer.decode([tid]) for tid in input_ids]
    full_decoded = "".join(token_texts)

    char_start = full_decoded.find(text)
    if char_start == -1:
        raise ValueError(
            f"Text not found in decoded prompt. text={text!r}, "
            f"decoded_start={full_decoded[:300]!r}"
        )

    cursor = 0
    for token_idx, token_text in enumerate(token_texts):
        token_end = cursor + len(token_text)
        if token_end > char_start:
            return token_idx
        cursor = token_end

    raise ValueError("No token index matched the text span.")


def build_trial_prompt(tokenizer) -> tuple[str, torch.Tensor, int]:
    """Build the tokenized trial prompt and locate the injection start index.

    Returns:
        formatted_prompt: The chat-templated prompt string.
        input_ids: Token IDs as a tensor of shape (1, seq_len).
        injection_start_idx: Token index where injection should begin
            (the double-newline before "Trial 1").
    """
    messages = build_messages()

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = tokenizer(formatted_prompt, return_tensors="pt")["input_ids"]

    injection_start_idx = find_first_token_index(
        input_ids[0].tolist(), tokenizer, INJECTION_BOUNDARY
    )

    return formatted_prompt, input_ids, injection_start_idx


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from transformers import AutoTokenizer

    MODEL_ID = "meta-llama/Llama-3.1-70B-Instruct"
    print(f"Loading tokenizer for {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    prompt, ids, start_idx = build_trial_prompt(tok)
    seq_len = ids.shape[1]

    print(f"\nFormatted prompt ({seq_len} tokens):\n")
    print(prompt)
    print(f"\n--- Injection starts at token {start_idx} / {seq_len} ---")

    # Show tokens around the injection boundary
    window = 5
    lo = max(0, start_idx - window)
    hi = min(seq_len, start_idx + window)
    print(f"\nTokens [{lo}..{hi}):")
    for i in range(lo, hi):
        token_str = tok.decode([ids[0, i]])
        marker = " <-- injection start" if i == start_idx else ""
        print(f"  [{i:3d}] {token_str!r}{marker}")
