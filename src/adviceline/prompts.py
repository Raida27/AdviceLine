"""Build the benchmark items (data/prompts.csv) from the four source files in data/.

Design (300 items):
  * 240 single-turn items = 60 questions x 4 detail levels
      D0 question only
      D1 a few personal details + question
      D2 full persona + question
      D3 full persona + a vulnerability + question
  * 60 three-turn items: turn 1 is the D1 version, turns 2 and 3 push for a recommendation

Run:  python -m adviceline.prompts
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from adviceline.config import DATA


def load_sources(data_dir: Path = DATA) -> dict[str, pd.DataFrame]:
    """Read the four hand-written source tables."""
    return {
        "questions": pd.read_csv(data_dir / "questions.csv"),
        "personas": pd.read_csv(data_dir / "personas.csv"),
        "vulnerabilities": pd.read_csv(data_dir / "vulnerabilities.csv"),
        "pressure": pd.read_csv(data_dir / "pressure.csv"),
    }


def build_items(src: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine questions, personas, vulnerabilities and pressure follow-ups into items."""
    questions = src["questions"].merge(src["personas"], on="topic", how="left")
    if questions["persona_id"].isna().any():
        missing = questions.loc[questions["persona_id"].isna(), "topic"].unique()
        raise ValueError(f"No persona for topics: {list(missing)}")

    vulns = src["vulnerabilities"].to_dict("records")
    pressure = src["pressure"]
    turn2 = pressure[pressure["turn"] == 2]["text"].tolist()
    turn3 = pressure[pressure["turn"] == 3]["text"].tolist()

    rows = []
    for i, q in enumerate(questions.to_dict("records")):
        # Rotate through the vulnerabilities and pressure messages so each is used evenly
        vuln = vulns[i % len(vulns)]
        texts = {
            0: q["question"],
            1: f"{q['light']} {q['question']}",
            2: f"{q['full']} {q['question']}",
            3: f"{q['full']} {vuln['text']} {q['question']}",
        }
        base = {
            "question_id": q["question_id"],
            "topic": q["topic"],
            "ask_type": q["ask_type"],
            "persona_id": q["persona_id"],
        }
        for level, text in texts.items():
            rows.append({
                "item_id": f"S-{q['question_id']}-D{level}",
                "kind": "single",
                **base,
                "detail_level": level,
                "vuln_id": vuln["vuln_id"] if level == 3 else "",
                "turn1": text,
                "turn2": "",
                "turn3": "",
            })
        rows.append({
            "item_id": f"M-{q['question_id']}",
            "kind": "multi",
            **base,
            "detail_level": 1,
            "vuln_id": "",
            "turn1": texts[1],
            "turn2": turn2[i % len(turn2)],
            "turn3": turn3[(i // len(turn2)) % len(turn3)],
        })
    return pd.DataFrame(rows)


def validate(items: pd.DataFrame) -> None:
    """Fail loudly if the design is broken."""
    assert items["item_id"].is_unique, "item_id must be unique"
    assert (items["kind"] == "single").sum() == 240, "expected 240 single-turn items"
    assert (items["kind"] == "multi").sum() == 60, "expected 60 multi-turn items"
    assert items["turn1"].str.len().min() > 10, "an item has an empty first message"
    multi = items[items["kind"] == "multi"]
    assert (multi["turn2"] != "").all() and (multi["turn3"] != "").all()


def turns(item: dict) -> list[str]:
    """The user messages of one item, in order."""
    return [t for t in (item["turn1"], item.get("turn2", ""), item.get("turn3", "")) if isinstance(t, str) and t]


def main() -> None:
    items = build_items(load_sources())
    validate(items)
    out = DATA / "prompts.csv"
    items.to_csv(out, index=False)
    print(f"Wrote {len(items)} items to {out}")
    print(items.groupby(["kind", "ask_type"]).size().to_string())


if __name__ == "__main__":
    main()
