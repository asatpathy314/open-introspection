"""Statistical utilities for the steering evaluation harness."""

import numpy as np
from scipy import stats


def bootstrap_ci(data: list[float], n_boot: int = 1000, ci: float = 0.95) -> tuple[float, float, float]:
    """Bootstrap mean and confidence interval. Returns (mean, ci_low, ci_high)."""
    data = np.array(data)
    if len(data) == 0:
        return 0.0, 0.0, 0.0
    boot_means = np.array([
        np.mean(np.random.choice(data, size=len(data), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return float(np.mean(data)), float(np.percentile(boot_means, 100 * alpha)), float(np.percentile(boot_means, 100 * (1 - alpha)))


def steerability_significance(steerabilities: list[float]) -> tuple[float, float]:
    """One-sided t-test: H0: mean steerability <= 0. Returns (t_stat, p_value)."""
    arr = np.array(steerabilities)
    if len(arr) < 2:
        return 0.0, 1.0
    t_stat, p_two = stats.ttest_1samp(arr, 0)
    p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
    return float(t_stat), float(p_one)


def benjamini_hochberg(p_values: list[float], q: float = 0.05) -> list[bool]:
    """
    Benjamini-Hochberg FDR correction.
    Returns list of bools: True = significant after correction.
    """
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    significant = [False] * n
    # Find largest k where p_(k) <= k/n * q
    max_k = -1
    for rank, (orig_idx, p) in enumerate(indexed, 1):
        if p <= rank / n * q:
            max_k = rank
    # All with rank <= max_k are significant
    if max_k > 0:
        for rank, (orig_idx, p) in enumerate(indexed, 1):
            if rank <= max_k:
                significant[orig_idx] = True
    return significant


def binomial_test(successes: int, trials: int, chance: float = 0.1) -> float:
    """One-sided binomial test: P(X >= successes) under H0: p = chance."""
    if trials == 0:
        return 1.0
    return float(stats.binomtest(successes, trials, chance, alternative="greater").pvalue)


def spearman_monotonicity(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Spearman rank correlation for monotonicity. Returns (rho, p_value)."""
    if len(xs) < 3:
        return 0.0, 1.0
    rho, p = stats.spearmanr(xs, ys)
    return float(rho), float(p)
