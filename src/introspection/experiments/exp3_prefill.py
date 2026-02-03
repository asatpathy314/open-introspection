"""Experiment 3: Detecting Unintended Outputs (Prefill Detection).

Can the model detect when its initial response was artificially prefilled
rather than generated intentionally?

Procedure per trial:
1. Ask model for a word association
2. Force the model's initial response to be a specific word (prefill)
3. Ask follow-up: "Did you intentionally choose that word?"
4. Model should detect prefill as not intentional
"""

from __future__ import annotations

from introspection.evaluation.results import TrialResult
from introspection.experiments.base import BaseExperiment
from introspection.prompts.templates import PromptBuilder
from introspection.vectors.dataset import ConceptWord


class PrefillDetectionExperiment(BaseExperiment):
    """Experiment 3: Can the model detect artificial prefills?"""

    experiment_name = "prefill_detect"

    def run_single_trial(
        self,
        concept: ConceptWord,
        layer: int,
        alpha: float,
        token_position: str,
    ) -> TrialResult:
        temperature = self._config.get("temperature", 1.0)
        tokenizer = self._model.get_tokenizer()

        # 1. Build the word association prompt
        messages = PromptBuilder.build_prefill_prompt()
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 2. Generate with forced prefill (the concept word)
        prefill_text = concept.word
        response = self._injector.generate_with_prefill(
            prompt,
            prefill_text,
            injection=None,  # No vector injection for this experiment
            max_new_tokens=self._model.config.max_new_tokens,
            temperature=temperature,
        )

        # 3. Build follow-up asking if it was intentional
        followup_messages = PromptBuilder.build_prefill_followup_prompt()
        followup_prompt = tokenizer.apply_chat_template(
            followup_messages, tokenize=False, add_generation_prompt=True
        )

        followup_response = self._injector.generate_clean(
            followup_prompt,
            max_new_tokens=self._model.config.max_new_tokens,
            temperature=temperature,
        )

        # 4. Evaluate: did the model detect the prefill?
        full_response = f"Word association: {response}\n\nFollow-up: {followup_response}"
        judge_results = self._judge.evaluate_trial(
            "prefill_detect", followup_response, concept.word
        )

        success = all(jr.verdict for jr in judge_results.values())

        return TrialResult(
            experiment=self.experiment_name,
            concept_word=concept.word,
            layer=layer,
            alpha=alpha,
            model_response=full_response,
            judge_results=judge_results,
            success=success,
            temperature=temperature,
            token_position=token_position,
            seed=self._config.get("seed"),
        )
