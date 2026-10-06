"""Fine-tune a small transformer to predict the rubric level (needs a GPU: run it in Colab).

It trains on LLM-judge labels ("silver") and tests on answers to questions it has never seen.
If you have labelled answers yourself, it also tests on those ("gold"); it never trains on them.

Run:  python -m adviceline.finetune --results-dir /content/drive/MyDrive/AdviceLine/results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

from adviceline.config import RESULTS
from adviceline.db import build
from adviceline.metrics import cohen_kappa


def scores(y, pred) -> dict:
    return {
        "n": int(len(y)),
        "accuracy": round(accuracy_score(y, pred), 3),
        "macro_f1": round(f1_score(y, pred, average="macro"), 3),
        "kappa": round(cohen_kappa(y, pred, labels=[0, 1, 2, 3]), 3),
        "kappa_quadratic": round(cohen_kappa(y, pred, weights="quadratic", labels=[0, 1, 2, 3]), 3),
    }


def finetune_classifier(model_id: str = "answerdotai/ModernBERT-base", epochs: int = 3, lr: float = 3e-5,
                        batch_size: int = 16, max_length: int = 512, results_dir: Path = RESULTS) -> dict:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    results = Path(results_dir)
    con = build(results_dir=results)
    judge = con.execute("SELECT response_id, question_id, response, level FROM labelled_judge").df()
    human = con.execute("SELECT response_id, response, level FROM labelled_human").df()
    judge = judge[~judge["response_id"].isin(human["response_id"])]  # never train on answers you labelled
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train_idx, test_idx = next(split.split(judge, groups=judge["question_id"]))
    train, test = judge.iloc[train_idx], judge.iloc[test_idx]
    print(f"train {len(train)} answers, test {len(test)} answers on unseen questions, human-labelled {len(human)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=4).to(device)

    # Rare levels count more in the loss, so the model doesn't just predict the most common level
    counts = np.bincount(train["level"].astype(int), minlength=4) + 1
    class_weights = torch.tensor(len(train) / (4 * counts), dtype=torch.float, device=device)

    def encode(texts):
        enc = tok(list(texts), truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        return {k: v.to(device) for k, v in enc.items()}

    def predict(df: pd.DataFrame) -> np.ndarray:
        model.eval()
        probs = []
        with torch.no_grad():
            for s in range(0, len(df), batch_size * 2):
                with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                    logits = model(**encode(df["response"].iloc[s:s + batch_size * 2])).logits
                probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        return np.concatenate(probs)

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * int(np.ceil(len(train) / batch_size))
    schedule = get_linear_schedule_with_warmup(optimiser, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    for epoch in range(epochs):
        model.train()
        order = np.random.default_rng(epoch).permutation(len(train))
        total = 0.0
        for s in range(0, len(train), batch_size):
            part = train.iloc[order[s:s + batch_size]]
            labels = torch.tensor(part["level"].astype(int).to_numpy(), device=device)
            with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                logits = model(**encode(part["response"])).logits
            loss = F.cross_entropy(logits.float(), labels, weight=class_weights)
            optimiser.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            schedule.step()
            total += loss.item() * len(part)
        test_pred = predict(test).argmax(axis=1)
        print(f"epoch {epoch + 1}: train loss {total / len(train):.3f} | unseen questions {scores(test['level'].astype(int), test_pred)}")

    report = {"model_id": model_id, "trained_on": "judge labels",
              "test_unseen_questions_vs_judge": scores(test["level"].astype(int), predict(test).argmax(axis=1))}
    if len(human):
        report["test_vs_human_labels"] = scores(human["level"].astype(int), predict(human).argmax(axis=1))

    out_dir = results / "models" / "transformer"
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    (results / "tables").mkdir(parents=True, exist_ok=True)
    (results / "tables" / "classifier_transformer.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", default="answerdotai/ModernBERT-base")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--results-dir", default=str(RESULTS))
    args = p.parse_args()
    finetune_classifier(args.model_id, args.epochs, args.lr, args.batch_size, args.max_length, Path(args.results_dir))


if __name__ == "__main__":
    main()
