"""LLM judge for evaluating model responses via OpenAI API.

Each evaluation criterion has a YES/NO prompt sent to gpt-5-nano.
Results are parsed and returned as structured JudgeResult objects.
"""

from __future__ import annotations

import logging
import time
from enum import Enum

import openai

from introspection.prompts.templates import PromptBuilder

logger = logging.getLogger(__name__)


class JudgeCriteria(Enum):
    """Evaluation criteria from the introspection paper."""

    COHERENCE = "coherence"
    AFFIRMATIVE = "affirmative"
    CORRECT_ID = "correct_identification"
    THINKING_ABOUT = "thinking_about_word"
    MCQ_CORRECT = "mcq_correct"
    PREFILL_DETECTED = "prefill_detected"


class JudgeResult:
    """Result of a single judge evaluation."""

    def __init__(self, criterion: JudgeCriteria, verdict: bool, raw_response: str) -> None:
        self.criterion = criterion
        self.verdict = verdict
        self.raw_response = raw_response

    def to_dict(self) -> dict:
        return {
            "criterion": self.criterion.value,
            "verdict": self.verdict,
            "raw_response": self.raw_response,
        }


# Map experiment names to their required criteria
EXPERIMENT_CRITERIA: dict[str, list[JudgeCriteria]] = {
    "self_report": [
        JudgeCriteria.COHERENCE,
        JudgeCriteria.AFFIRMATIVE,
        JudgeCriteria.CORRECT_ID,
    ],
    "distinguish": [
        JudgeCriteria.COHERENCE,
        JudgeCriteria.MCQ_CORRECT,
    ],
    "prefill_detect": [
        JudgeCriteria.COHERENCE,
        JudgeCriteria.PREFILL_DETECTED,
    ],
    "intentional_control": [
        JudgeCriteria.COHERENCE,
        JudgeCriteria.THINKING_ABOUT,
    ],
}


class LLMJudge:
    """Evaluates model responses using OpenAI API."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5-nano",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    def _get_judge_prompt(
        self,
        criterion: JudgeCriteria,
        model_response: str,
        concept_word: str = "",
        **kwargs: str,
    ) -> str:
        """Build the judge prompt for a specific criterion."""
        if criterion == JudgeCriteria.COHERENCE:
            return PromptBuilder.JUDGE_COHERENCE.format(response=model_response)
        elif criterion == JudgeCriteria.AFFIRMATIVE:
            return PromptBuilder.JUDGE_AFFIRMATIVE.format(response=model_response)
        elif criterion == JudgeCriteria.CORRECT_ID:
            return PromptBuilder.JUDGE_CORRECT_ID.format(
                response=model_response, word=concept_word
            )
        elif criterion == JudgeCriteria.THINKING_ABOUT:
            return PromptBuilder.JUDGE_THINKING_ABOUT.format(
                response=model_response, word=concept_word
            )
        elif criterion == JudgeCriteria.MCQ_CORRECT:
            return PromptBuilder.JUDGE_MCQ.format(
                response=model_response,
                correct=concept_word,
                options=kwargs.get("options", ""),
            )
        elif criterion == JudgeCriteria.PREFILL_DETECTED:
            return PromptBuilder.JUDGE_PREFILL_DETECTED.format(response=model_response)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

    def evaluate(
        self,
        criterion: JudgeCriteria,
        model_response: str,
        concept_word: str = "",
        **kwargs: str,
    ) -> JudgeResult:
        """Evaluate a model response against a single criterion.

        Sends a YES/NO prompt to the judge model and parses the response.
        Retries with exponential backoff on failure.

        Returns:
            JudgeResult with verdict (bool) and raw response.
        """
        prompt = self._get_judge_prompt(criterion, model_response, concept_word, **kwargs)

        for attempt in range(self._max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0,
                )
                raw = response.choices[0].message.content.strip().upper()

                # Parse YES/NO
                verdict = raw.startswith("YES")

                return JudgeResult(
                    criterion=criterion,
                    verdict=verdict,
                    raw_response=raw,
                )

            except (openai.APIError, openai.RateLimitError) as e:
                if attempt < self._max_retries - 1:
                    delay = self._retry_delay * (2**attempt)
                    logger.warning(
                        "Judge API error (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        self._max_retries,
                        delay,
                        str(e),
                    )
                    time.sleep(delay)
                else:
                    logger.error("Judge API failed after %d attempts", self._max_retries)
                    raise

        # Should not reach here, but return a failure result just in case
        return JudgeResult(criterion=criterion, verdict=False, raw_response="ERROR")

    def evaluate_trial(
        self,
        experiment: str,
        model_response: str,
        concept_word: str = "",
        **kwargs: str,
    ) -> dict[JudgeCriteria, JudgeResult]:
        """Run all relevant criteria for a given experiment type.

        Args:
            experiment: Experiment name (self_report, distinguish, etc.)
            model_response: The model's raw response text.
            concept_word: The target concept word.
            **kwargs: Additional args (e.g., options for MCQ).

        Returns:
            Dict mapping each criterion to its JudgeResult.
        """
        criteria = EXPERIMENT_CRITERIA.get(experiment, [])
        if not criteria:
            raise ValueError(
                f"Unknown experiment '{experiment}'. "
                f"Available: {list(EXPERIMENT_CRITERIA.keys())}"
            )

        results = {}
        for criterion in criteria:
            results[criterion] = self.evaluate(
                criterion, model_response, concept_word, **kwargs
            )

        return results
