"""
This file contains the main code for LLM judging.

Each output is passed through 4 rubrics.

1. Coherence.
2. Does the model describe itself as "thinking about the word"?
3. Does the model affirm that they noticed something unusal?
4. Does the model give an affirmative response AND identify the thought correctly in that order?
"""

import json
import time
from pathlib import Path

import anthropic
import prompt
from anthropic.types.messages.batch_create_params import Request
from dotenv import load_dotenv

MODEL = "claude-sonnet-4-6"

load_dotenv()
client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY implicitly


def load_trials(experiment_dirs: list[Path]) -> list[dict]:
    # TODO: for each dir load the trials from the `trials.jsonl` file
    raise NotImplementedError()


def build_batch_requests(trials: list[dict], prompt_template: str) -> list[Request]:
    """
    As a note the prompt template must have `{question}` and `{response}` placeholders.

    We set temperature to 0 to ensure deterministic output.
    """
    return [
        {
            "custom_id": str(idx),
            "params": {
                "max_tokens": 256,
                "temperature": 0,
                "messages": [
                    {
                        "content": prompt_template.format(
                            question=t["question"], response=t["response"]
                        ),
                        "role": "user",
                    }
                ],
                "model": MODEL,
            },
        }
        for idx, t in enumerate(trials)
    ]


def grade_trials(trials: list[dict]) -> list[dict]:
    prompt_templates = [
        prompt.COHERENCE_PROMPT,
        prompt.THINKING_ABOUT_WORD_PROMPT,
        prompt.AFFIRMATIVE_RESPONSE_PROMPT,
        prompt.AFFIRMATIVE_WITH_IDENTIFICATION_PROMPT,
    ]

    batches = []

    for prompt_template in prompt_templates:
        batch_requests = build_batch_requests(trials, prompt_template)
        batch = client.messages.batches.create(requests=batch_requests)
        batches.append(batch)

    while True:
        for batch in batches:
            # TODO: check the status of each batch use the docs https://platform.claude.com/docs/en/build-with-claude/batch-processing
            # while batches not finished, wait
            pass

    return trials


if __name__ == "__main__":
    # TODO: print a summary of what trials were successfully analyzed, and which ones weren't

    # TODO: create a function to do this. Create a new file with all analyzed trials + labels in an analysis folder with a sensible default + a metadata folder that contains prompts and model data + batch data.
    #
