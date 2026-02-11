"""Token span detection for prefill-detection experiment."""


def find_sentence_token_indices(
    input_ids: list[int],
    tokenizer,
    sentence: str,
) -> list[int]:
    """
    Given fully tokenized input from apply_chat_template, find the
    token indices corresponding to `sentence`.

    Strategy: decode tokens, find the sentence in decoded text,
    then linearly collect tokens whose character ranges overlap
    that sentence span. Validate by decoding the span.

    Returns: sorted list of integer indices into input_ids.
    """
    if not sentence:
        raise ValueError("Sentence must be non-empty.")
    if not input_ids:
        raise ValueError("input_ids must be non-empty.")

    # Step 1: decode each token and reconstruct decoded prompt text.
    token_texts: list[str] = []
    for tid in input_ids:
        token_texts.append(tokenizer.decode([tid]))
    full_text = "".join(token_texts)

    # Step 2: find sentence in the full decoded text
    char_start = full_text.find(sentence)
    if char_start == -1:
        raise ValueError(
            f"Sentence not found in decoded prompt.\n"
            f"  Sentence: {sentence!r}\n"
            f"  Decoded prompt (first 500 chars): {full_text[:500]!r}"
        )
    char_end = char_start + len(sentence)

    # Step 3: map character span to overlapping token indices.
    token_indices: list[int] = []
    cursor = 0
    for token_idx, token_text in enumerate(token_texts):
        token_start = cursor
        token_end = token_start + len(token_text)
        cursor = token_end

        if token_end <= char_start:
            continue
        if token_start >= char_end:
            break
        token_indices.append(token_idx)

    # Step 4: decode the span and check it contains the sentence
    span_ids = [input_ids[i] for i in token_indices]
    decoded_span = tokenizer.decode(span_ids)

    if sentence not in decoded_span:
        raise ValueError(
            f"Validation failed: decoded span does not contain sentence.\n"
            f"  Sentence: {sentence!r}\n"
            f"  Span text: {decoded_span!r}\n"
            f"  Token indices: {token_indices}"
        )

    return token_indices
