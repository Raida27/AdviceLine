"""Predict the rubric level of an answer.

Baseline here: TF-IDF + logistic regression (runs on a laptop in seconds).
Stronger model: a fine-tuned small transformer (notebooks/02_finetune_classifier.ipynb, Colab GPU).

Evaluation uses GroupKFold by question: answers to the same question never sit in both the
training and the test fold. Without this, the model can memorise questions and look better than it is.

Run:
    python -m adviceline.classify --labels judge
    python -m adviceline.classify --labels human --who raida
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from adviceline.config import BASELINE_MODEL_PATH, DATA, DB_PATH, RESULTS
from adviceline.db import build, use_labels
from adviceline.metrics import cohen_kappa, expected_calibration_error
from adviceline.rules import detect_advice


def make_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])


def load_labelled(source: str = "judge", who: str | None = None, results_dir: Path = RESULTS,
                  data_dir: Path = DATA, db_path: Path = DB_PATH) -> pd.DataFrame:
    con = build(results_dir=results_dir, data_dir=data_dir, db_path=db_path)
    use_labels(con, source, who)
    return con.execute("SELECT response_id, question_id, response, level FROM labelled").df()


def evaluate(df: pd.DataFrame, n_splits: int = 5) -> dict:
    """Cross-validated scores for the baseline and the rule-based detector on the same answers."""
    X, y, groups = df["response"], df["level"].astype(int).to_numpy(), df["question_id"]
    n_splits = min(n_splits, groups.nunique())
    if n_splits < 2:
        raise ValueError("Need answers to at least 2 different questions.")
    proba = cross_val_predict(make_pipeline(), X, y, groups=groups, cv=GroupKFold(n_splits),
                              method="predict_proba")
    classes = np.unique(y)
    pred = classes[proba.argmax(axis=1)]

    # Binary view used by the guardrail: is it beyond guidance (level 2 or 3)?
    p_beyond = proba[:, classes >= 2].sum(axis=1)
    y_beyond = (y >= 2).astype(int)
    rule_levels = np.array([detect_advice(t)["level"] for t in X])

    report = {
        "n_answers": int(len(df)),
        "n_questions": int(groups.nunique()),
        "label_counts": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "baseline": {
            "accuracy": round(accuracy_score(y, pred), 3),
            "macro_f1": round(f1_score(y, pred, average="macro"), 3),
            "kappa": round(cohen_kappa(y, pred, labels=[0, 1, 2, 3]), 3),
            "kappa_quadratic": round(cohen_kappa(y, pred, weights="quadratic", labels=[0, 1, 2, 3]), 3),
            "beyond_guidance_brier": round(brier_score_loss(y_beyond, p_beyond), 3),
            "beyond_guidance_ece": round(expected_calibration_error(p_beyond, y_beyond), 3),
        },
        "rules": {
            "accuracy": round(accuracy_score(y, rule_levels), 3),
            "macro_f1": round(f1_score(y, rule_levels, average="macro"), 3),
            "kappa": round(cohen_kappa(y, rule_levels, labels=[0, 1, 2, 3]), 3),
        },
    }
    if len(set(y_beyond)) == 2:
        report["baseline"]["beyond_guidance_auc"] = round(roc_auc_score(y_beyond, p_beyond), 3)
    return report


def train(df: pd.DataFrame, path: Path = BASELINE_MODEL_PATH) -> Pipeline:
    model = make_pipeline().fit(df["response"], df["level"].astype(int))
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return model


def run_classifier(source: str = "judge", who: str | None = None, results_dir: Path = RESULTS) -> dict:
    """Evaluate the baseline with cross-validation, train it on all labels, save the model and report."""
    df = load_labelled(source, who, results_dir=results_dir)
    if df.empty:
        raise SystemExit("No labelled answers yet. Run the judge (Colab) or label some answers first.")
    report = evaluate(df)
    train(df, Path(results_dir) / "models" / "baseline.joblib")
    out = Path(results_dir) / "tables" / f"classifier_{source}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"Saved the model and the report ({out})")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", choices=["judge", "human"], default="judge")
    p.add_argument("--who", default=None)
    args = p.parse_args()
    print(json.dumps(run_classifier(args.labels, args.who), indent=2))


if __name__ == "__main__":
    main()
