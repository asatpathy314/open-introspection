"""Concept word dataset for vector extraction and experiments."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ConceptWord:
    """A single concept word with its category."""

    word: str
    category: str  # "abstract", "concrete", "person", "country", "verb"


@dataclass
class ConceptDataset:
    """The full dataset of concept words and baseline words."""

    concepts: list[ConceptWord]
    baselines: list[str]

    @classmethod
    def load(cls, path: str | Path) -> ConceptDataset:
        """Load concept dataset from JSON file.

        Expected format:
        {
            "concepts": [{"word": "justice", "category": "abstract"}, ...],
            "baselines": ["desk", "jacket", ...]
        }
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        concepts = [ConceptWord(word=c["word"], category=c["category"]) for c in data["concepts"]]
        baselines = data["baselines"]

        return cls(concepts=concepts, baselines=baselines)

    def sample_concept(self, rng: random.Random) -> ConceptWord:
        """Sample a random concept word."""
        return rng.choice(self.concepts)

    def sample_distractors(
        self,
        exclude: str,
        n: int,
        rng: random.Random,
    ) -> list[str]:
        """Sample n distractor concept words, excluding the target.

        Used for MCQ evaluation in Experiment 2.
        """
        available = [c.word for c in self.concepts if c.word != exclude]
        return rng.sample(available, min(n, len(available)))

    def get_concept(self, word: str) -> ConceptWord | None:
        """Look up a concept by word."""
        for c in self.concepts:
            if c.word == word:
                return c
        return None

    @property
    def concept_words(self) -> list[str]:
        """All concept words as strings."""
        return [c.word for c in self.concepts]
