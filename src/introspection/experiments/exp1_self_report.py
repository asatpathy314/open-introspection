"""Experiment 1: Self-Report of Injected Thoughts.

Can the model detect and identify concepts that have been injected
into its residual stream activations?

Procedure per trial:
1. Sample a concept word
2. Load (or compute) concept vector for this word at the target layer
3. Build self-report prompt via PromptBuilder
4. Apply chat template to get the full formatted prompt string
5. Generate response with injection (Injector.generate_with_injection)
6. Evaluate with LLMJudge: COHERENCE + AFFIRMATIVE + CORRECT_ID
7. Success = all three criteria are YES

Expected: ~20% success rate under optimal conditions (layer ~2/3 depth, alpha 4-9)
for capable models. Open-source models may show 5-15%.
"""

from __future__ import annotations

from introspection.evaluation.results import TrialResult
from introspection.experiments.base import BaseExperiment
from introspection.injection.injector import InjectionConfig
from introspection.prompts.templates import PromptBuilder
from introspection.vectors.dataset import ConceptWord


class SelfReportExperiment(BaseExperiment):
    """Experiment 1: Can the model detect and identify injected concepts?"""

    experiment_name = "self_report"

    def run_single_trial(
        self,
        concept: ConceptWord,
        layer: int,
        alpha: float,
        token_position: str,
    ) -> TrialResult:
        # 1. Get concept vector (from cache or compute)
        vec = self._vectors.cache.get(concept.word, layer, token_position)
        if vec is None:
            vec = self._vectors.compute_concept_vector(
                concept.word,
                self._dataset.baselines,
                layer,
                token_position,
            )

        # 2. Build injection config
        injection = InjectionConfig(
            layer=layer,
            alpha=alpha,
            vector=vec,
            apply_to=self._config.get("apply_to", "all"),
        )

        # 3. Build prompt and apply chat template
        messages = PromptBuilder.build_self_report_prompt()
        tokenizer = self._model.get_tokenizer()
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 4. Generate with injection
        temperature = self._config.get("temperature", 1.0)
        max_new_tokens = self._model.config.max_new_tokens
        response = self._injector.generate_with_injection(
            prompt,
            injection,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

        # 5. Evaluate with LLM judge
        judge_results = self._judge.evaluate_trial(
            "self_report", response, concept.word
        )

        # 6. Success = all criteria pass
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
