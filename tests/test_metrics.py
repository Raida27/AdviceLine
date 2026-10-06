import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import cohen_kappa_score

from adviceline.metrics import (cluster_bootstrap, cohen_kappa, expected_calibration_error, kappa_label,
                                mcnemar_exact, two_proportion_z, wilson_ci)


def test_wilson_matches_known_values():
    p, lo, hi = wilson_ci(5, 10)
    assert p == 0.5
    assert lo == pytest.approx(0.2366, abs=1e-4)
    assert hi == pytest.approx(0.7634, abs=1e-4)
    p, lo, hi = wilson_ci(0, 10)
    assert lo == 0 and hi == pytest.approx(0.2775, abs=1e-4)


@pytest.mark.parametrize("weights", [None, "linear", "quadratic"])
def test_kappa_matches_scikit_learn(weights):
    rng = np.random.default_rng(1)
    a = rng.integers(0, 4, 200)
    b = np.where(rng.random(200) < 0.7, a, rng.integers(0, 4, 200))
    assert cohen_kappa(a, b, weights=weights) == pytest.approx(cohen_kappa_score(a, b, weights=weights))


def test_kappa_perfect_and_label_wording():
    assert cohen_kappa([0, 1, 2, 3], [0, 1, 2, 3]) == 1.0
    assert kappa_label(0.85) == "almost perfect"
    assert kappa_label(0.5) == "moderate"


def test_mcnemar():
    # 5 answers became advice-like, none went the other way: p = 2 * (1/2)^5
    result = mcnemar_exact([0] * 5 + [1] * 3, [1] * 5 + [1] * 3)
    assert result["no_to_yes"] == 5 and result["yes_to_no"] == 0
    assert result["p_value"] == pytest.approx(0.0625)
    assert mcnemar_exact([1, 0], [1, 0])["p_value"] == 1.0


def test_two_proportion_z():
    assert two_proportion_z(10, 100, 10, 100)["p_value"] == pytest.approx(1.0)
    assert two_proportion_z(30, 100, 10, 100)["p_value"] < 0.001


def test_calibration_error():
    probs = np.array([0.1] * 10 + [0.9] * 10)
    outcomes = np.array([1] + [0] * 9 + [1] * 9 + [0])
    assert expected_calibration_error(probs, outcomes) == pytest.approx(0.0)
    assert expected_calibration_error([0.9] * 10, [0] * 10) == pytest.approx(0.9)


def test_cluster_bootstrap_interval_contains_estimate():
    df = pd.DataFrame({"q": np.repeat(range(20), 4), "y": np.tile([0, 0, 1, 1], 20)})
    est, lo, hi = cluster_bootstrap(df, "y", "q", n_boot=300)
    assert lo <= est <= hi and est == 0.5
