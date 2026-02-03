"""Experiment 2: Distinguishing Thoughts from Text.

Can the model distinguish between concepts injected into its activations
("thoughts") and concepts presented as text input?

Procedure per trial:
1. Sample a concept word + 9 distractors (10-word MCQ)
2. Two conditions on the same trial:
   a. "Thought" condition: Inject concept vector, present MCQ
   b. "Text" condition: Show word in text, present same MCQ
3. LLM judge checks if model selected the correct word
4. Success: judge picks correct word (baseline = 10% by chance)
"""

from __future__ import annotations

from introspection.evaluation.results import TrialResult
from introspection.experiments.base import BaseExperiment
from introspection.injection.injector import InjectionConfig
from introspection.prompts.templates import PromptBuilder
from introspection.vectors.dataset import ConceptWord


class DistinguishExperiment(BaseExperiment):
    """Experiment 2: Can the model tell thoughts from text?"""

    experiment_name = "distinguish"

    def run_single_trial(
        self,
        concept: ConceptWord,
        layer: int,
        alpha: float,
        token_position: str,
    ) -> TrialResult:
        num_distractors = self._config.get("num_distractors", 9)
        temperature = self._config.get("temperature", 1.0)

        # 1. Get concept vector
        vec = self._vectors.cache.get(concept.word, layer, token_position)
        if vec is None:
            vec = self._vectors.compute_concept_vector(
                concept.word,
                self._dataset.baselines,
                layer,
                token_position,
            )

        # 2. Sample distractors and build MCQ prompt
        distractors = self._dataset.sample_distractors(
            concept.word, num_distractors, self._rng
        )
        messages, options = PromptBuilder.build_distinguish_think_prompt(
            concept.word, distractors, self._rng
        )

        # 3. Build injection config
        injection = InjectionConfig(
            layer=layer,
            alpha=alpha,
            vector=vec,
            apply_to=self._config.get("apply_to", "all"),
        )

        # 4. Apply chat template and generate with injection
        tokenizer = self._model.get_tokenizer()
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        response = self._injector.generate_with_injection(
            prompt,
            injection,
            max_new_tokens=self._model.config.max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

        # 5. Evaluate with LLM judge
        options_str = ", ".join(options)
        judge_results = self._judge.evaluate_trial(
            "distinguish",
            response,
            concept.word,
            options=options_str,
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
