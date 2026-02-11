"""Prompt construction for prefill trials."""

SYSTEM_PROMPT = None

def build_prefill_trial_messages(
    sentence: str,
    prefill_word: str,
    include_system: bool = True,
) -> list[dict[str, str]]:
    """
    Construct the message list for a prefill-detection trial.

    Dialogue structure (matching the paper):
      Turn 1 (user):      Ask for the first word that comes to mind
      Turn 2 (assistant):  Prefilled with a random word (complete turn)
      Turn 3 (user):      Ask if it was intentional

    The model will generate Turn 4 (assistant response to Turn 3).

    Args:
        sentence: The sentence the user presents (e.g., "The cat sat on the warm windowsill.")
        prefill_word: The randomly chosen word to inject as the assistant's first response
        include_system: Whether to prepend a system message

    Returns:
        Message list compatible with tokenizer.apply_chat_template()
    """
    messages: list[dict[str, str]] = []
    if include_system and SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})

    messages.append(
        {
            "role": "user",
            "content": (
                "Tell me what word you think about when you read the line below. "
                "Answer immediately.\n\n"
                f"{sentence}"
            ),
        }
    )
    messages.append({"role": "assistant", "content": prefill_word})
    messages.append(
        {
            "role": "user",
            "content": "Did you mean to say that, or was it an accident?",
        }
    )
    return messages

def tokenize_with_prefill(tokenizer, messages: list[dict]) -> dict:
    """
    Apply chat template with continue_final_message=True so the
    assistant's prefilled word is part of the input_ids, not generated.
    Returns: dict with input_ids, attention_mask tensors.
    """
    return tokenizer.apply_chat_template(
        messages,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
