"""Experiment 4: Intentional Control of Internal States.

Can the model intentionally modulate its own representations
when instructed to "think about" a concept?

Procedure per trial:
1. Instruct model: "Think about {word} while you write the next sentence."
2. Generate response
3. Evaluate whether the model's response shows awareness of the concept
"""

from __future__ import annotations

from introspection.evaluation.results import TrialResult
from introspection.experiments.base import BaseExperiment
from introspection.prompts.templates import PromptBuilder
from introspection.vectors.dataset import ConceptWord


class IntentionalControlExperiment(BaseExperiment):
    """Experiment 4: Can the model modulate its own representations?"""

    experiment_name = "intentional_control"

    def run_single_trial(
        self,
        concept: ConceptWord,
        layer: int,
        alpha: float,
        token_position: str,
    ) -> TrialResult:
        temperature = self._config.get("temperature", 1.0)
        tokenizer = self._model.get_tokenizer()

        # 1. Build the intentional control prompt
        messages = PromptBuilder.build_control_prompt(concept.word)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 2. Generate without injection (model tries to think about concept on its own)
        response = self._injector.generate_clean(
            prompt,
            max_new_tokens=self._model.config.max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

        # 3. Evaluate: does the response show thinking about the concept?
        judge_results = self._judge.evaluate_trial(
            "intentional_control", response, concept.word
        )

        success = all(jr.verdict for jr in judge_results.values())

        return TrialResult(
            experiment=self.experiment_name,
            concept_word=concept.word,
            layer=layer,
            alpha=alpha,
            model_response=response,
            judge_results=judge_results,
            success=success,
            temperature=temperature,
            token_position=token_position,
            seed=self._config.get("seed"),
        )
