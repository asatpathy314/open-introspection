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
    trials = []
    for d in experiment_dirs:
        trials_path = d / "trials.jsonl"
        with trials_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record["experiment_dir"] = str(d)
                trials.append(record)
    return trials


def build_batch_requests(trials: list[dict], prompt_template: str) -> list[Request]:
    """
    As a note the prompt template must have `{question}` and `{response}` placeholders.

    We set temperature to 0 to ensure deterministic output.
    """
    question = "\n\n".join(
        f"Human: {m['content']}" if m["role"] == "user" else f"Assistant: {m['content']}"
        for m in prompt.EXPERIMENT_MESSAGES
    )
    return [
        {
            "custom_id": str(idx),
            "params": {
                "max_tokens": 256,
                "temperature": 0,
                "messages": [
                    {
                        "content": prompt_template.format(
                            question=question,
                            response=t["response"],
                            word=t.get("concept", ""),
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
    rubric_names = [
        "coherence",
        "thinking_about_word",
        "affirmative_response",
        "affirmative_with_identification",
    ]

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
        print(f"Created batch {batch.id}")

    # Poll until all batches are done
    pending = set(range(len(batches)))
    time_start = time.time()
    while pending:
        time.sleep(5)
        for i in list(pending):
            batches[i] = client.messages.batches.retrieve(batches[i].id)
            if batches[i].processing_status == "ended":
                print(f"Batch {batches[i].id} ({rubric_names[i]}) ended")
                pending.discard(i)
        print(f"Pending: {len(pending)}")

    print(f"All batches ended in {time.time() - time_start:.1f}s")

    # Collect results
    for i, rubric_name in enumerate(rubric_names):
        for result in client.messages.batches.results(batches[i].id):
            trial_idx = int(result.custom_id)
            if result.result.type == "succeeded":
                text = "".join(
                    block.text for block in result.result.message.content
                    if block.type == "text"
                )
                trials[trial_idx][f"{rubric_name}_raw"] = text
                last_word = text.strip().split()[-1].strip(".,!?:;").upper()
                trials[trial_idx][rubric_name] = last_word if last_word in ("YES", "NO") else None
            else:
                trials[trial_idx][rubric_name] = None

    return trials


if __name__ == "__main__":
    # TODO: print a summary of what trials were successfully analyzed, and which ones weren't

    # TODO: create a function to do this. Create a new file with all analyzed trials + labels in an analysis folder with a sensible default + a metadata folder that contains prompts and model data + batch data.
    #
