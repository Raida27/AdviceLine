"""Statistics used in the analysis, written out so each formula is easy to explain.

* wilson_ci          confidence interval for a rate (e.g. 23 of 240 answers were advice-like)
* cluster_bootstrap  confidence interval when answers are grouped (several answers per question)
* cohen_kappa        agreement between two labellers beyond chance (unweighted, linear, quadratic)
* mcnemar_exact      paired before/after test (turn 1 vs turn 3, or answer vs guardrailed answer)
* two_proportion_z   compare two independent rates
* expected_calibration_error   do predicted probabilities match how often things happen?
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
import pandas as pd


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Rate k/n with a Wilson 95% interval. Better than p ± 1.96·SE when p is near 0 or 1."""
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return p, max(0.0, centre - half), min(1.0, centre + half)


def cluster_bootstrap(df: pd.DataFrame, value: str, cluster: str, stat: Callable = np.mean,
                      n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap interval that resamples whole clusters (e.g. questions), not single rows.

    Answers to the same question are related, so treating them as independent would make
    the interval too narrow.
    """
    rng = np.random.default_rng(seed)
    groups = [g[value].to_numpy() for _, g in df.groupby(cluster)]
    estimate = float(stat(df[value].to_numpy()))
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        boots.append(stat(np.concatenate([groups[i] for i in pick])))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return estimate, float(lo), float(hi)


def cohen_kappa(a: Sequence, b: Sequence, weights: str | None = None,
                labels: Sequence | None = None) -> float:
    """Cohen's kappa = (observed agreement - chance agreement) / (1 - chance agreement).

    weights=None counts any disagreement as total; "linear" or "quadratic" give partial credit
    for near misses, which suits an ordered scale like 0-3.
    """
    a, b = list(a), list(b)
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty label lists")
    labels = list(labels) if labels is not None else sorted(set(a) | set(b))
    index = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    observed = np.zeros((k, k))
    for x, y in zip(a, b):
        observed[index[x], index[y]] += 1
    observed /= observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))

    i, j = np.indices((k, k))
    if weights is None:
        w = (i != j).astype(float)
    elif weights == "linear":
        w = np.abs(i - j) / (k - 1)
    elif weights == "quadratic":
        w = ((i - j) / (k - 1)) ** 2
    else:
        raise ValueError("weights must be None, 'linear' or 'quadratic'")
    denom = (w * expected).sum()
    if denom == 0:
        return 1.0
    return float(1 - (w * observed).sum() / denom)


def kappa_label(kappa: float) -> str:
    """Landis & Koch (1977) wording for a kappa value."""
    for cut, word in [(0.0, "poor"), (0.2, "slight"), (0.4, "fair"), (0.6, "moderate"), (0.8, "substantial")]:
        if kappa <= cut:
            return word
    return "almost perfect"


def mcnemar_exact(before: Sequence[int], after: Sequence[int]) -> dict:
    """Exact McNemar test for paired yes/no outcomes.

    Only the discordant pairs matter: b = yes→no, c = no→yes. Under "no change", each
    discordant pair is a coin flip, so the p-value is a two-sided binomial test.
    """
    before, after = np.asarray(before, dtype=int), np.asarray(after, dtype=int)
    b = int(((before == 1) & (after == 0)).sum())
    c = int(((before == 0) & (after == 1)).sum())
    n = b + c
    if n == 0:
        p = 1.0
    else:
        tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2**n
        p = min(1.0, 2 * tail)
    return {"yes_to_no": b, "no_to_yes": c, "p_value": p}


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> dict:
    """z-test for the difference between two independent rates."""
    p1, p2 = k1 / n1, k2 / n2
    pooled = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))  # two-sided p-value from the normal distribution
    return {"diff": p1 - p2, "z": z, "p_value": p}


def expected_calibration_error(probs: Sequence[float], outcomes: Sequence[int], bins: int = 10) -> float:
    """Average gap between predicted probability and observed frequency, across probability bins."""
    probs, outcomes = np.asarray(probs, dtype=float), np.asarray(outcomes, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs > lo) & (probs <= hi) if lo > 0 else (probs >= lo) & (probs <= hi)
        if mask.any():
            ece += mask.mean() * abs(probs[mask].mean() - outcomes[mask].mean())
    return float(ece)
