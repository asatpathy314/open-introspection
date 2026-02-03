"""Experiment results persistence and loading."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from introspection.evaluation.judge import JudgeCriteria, JudgeResult

logger = logging.getLogger(__name__)


@dataclass
class TrialResult:
    """Result of a single experiment trial."""

    experiment: str
    concept_word: str
    layer: int
    alpha: float
    model_response: str
    judge_results: dict[JudgeCriteria, JudgeResult]
    success: bool
    temperature: float
    timestamp: str = ""
    token_position: str = "last"
    seed: int | None = None

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_flat_dict(self) -> dict[str, Any]:
        """Flatten for CSV/DataFrame export."""
        row: dict[str, Any] = {
            "experiment": self.experiment,
            "concept_word": self.concept_word,
            "layer": self.layer,
            "alpha": self.alpha,
            "success": self.success,
            "temperature": self.temperature,
            "token_position": self.token_position,
            "timestamp": self.timestamp,
            "seed": self.seed,
        }
        # Flatten judge results
        for criterion, result in self.judge_results.items():
            row[f"judge_{criterion.value}"] = result.verdict
        # Truncate response for CSV (keep full in JSONL)
        row["model_response_preview"] = self.model_response[:200]
        return row

    def to_jsonl_dict(self) -> dict[str, Any]:
        """Full serialization for JSONL storage."""
        return {
            "experiment": self.experiment,
            "concept_word": self.concept_word,
            "layer": self.layer,
            "alpha": self.alpha,
            "model_response": self.model_response,
            "judge_results": {
                k.value: v.to_dict() for k, v in self.judge_results.items()
            },
            "success": self.success,
            "temperature": self.temperature,
            "token_position": self.token_position,
            "timestamp": self.timestamp,
            "seed": self.seed,
        }


class ResultStore:
    """Manages experiment results persistence."""

    def __init__(self, output_dir: str | Path) -> None:
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _csv_path(self, experiment: str) -> Path:
        return self._dir / f"{experiment}.csv"

    def _jsonl_path(self, experiment: str) -> Path:
        return self._dir / f"{experiment}.jsonl"

    def save_trial(self, result: TrialResult) -> None:
        """Append a single trial result to both CSV and JSONL files."""
        # JSONL (full data)
        jsonl_path = self._jsonl_path(result.experiment)
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(result.to_jsonl_dict()) + "\n")

        # CSV (flattened summary)
        csv_path = self._csv_path(result.experiment)
        flat = result.to_flat_dict()
        file_exists = csv_path.exists()

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=flat.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(flat)

    def load_results(self, experiment: str) -> pd.DataFrame:
        """Load results CSV as a DataFrame."""
        csv_path = self._csv_path(experiment)
        if not csv_path.exists():
            return pd.DataFrame()
        return pd.read_csv(csv_path)

    def load_full_results(self, experiment: str) -> list[dict]:
        """Load full JSONL results."""
        jsonl_path = self._jsonl_path(experiment)
        if not jsonl_path.exists():
            return []
        results = []
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        return results

    def summary(self, experiment: str) -> dict[str, Any]:
        """Compute summary statistics for an experiment."""
        df = self.load_results(experiment)
        if df.empty:
            return {"total_trials": 0}

        summary: dict[str, Any] = {
            "total_trials": len(df),
            "success_rate": df["success"].mean(),
        }

        # Success rate by layer
        if "layer" in df.columns:
            summary["by_layer"] = (
                df.groupby("layer")["success"].mean().to_dict()
            )

        # Success rate by alpha
        if "alpha" in df.columns:
            summary["by_alpha"] = (
                df.groupby("alpha")["success"].mean().to_dict()
            )

        return summary
