"""Run every query in sql/analysis.sql, add confidence intervals and tests, and write a summary.

Run:
    python -m adviceline.analysis                    # LLM-judge labels ("silver")
    python -m adviceline.analysis --labels human     # your own labels ("gold")
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd

from adviceline.config import DATA, DB_PATH, RESULTS, ROOT
from adviceline.db import build, use_labels
from adviceline.metrics import cohen_kappa, kappa_label, mcnemar_exact, wilson_ci
from adviceline.rules import detect_advice

RATE_COLS = ["advice_like", "beyond_guidance", "refused", "signposted", "named_product"]


def load_queries(path: Path = ROOT / "sql" / "analysis.sql") -> dict[str, str]:
    """Split the SQL file into named queries ('-- name: by_model' starts a block)."""
    queries, name, lines = {}, None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith("-- name:"):
            if name:
                queries[name] = "\n".join(lines).strip()
            name, lines = line.split(":", 1)[1].strip(), []
        elif name:
            lines.append(line)
    if name:
        queries[name] = "\n".join(lines).strip()
    return queries


def add_rates(df: pd.DataFrame) -> pd.DataFrame:
    """For each count column, add the rate and its Wilson 95% interval."""
    df = df.copy()
    if "n" not in df:
        return df
    for col in RATE_COLS:
        if col not in df:
            continue
        cis = [wilson_ci(int(k), int(n)) for k, n in zip(df[col].fillna(0), df["n"])]
        df[f"{col}_rate"] = [round(p, 3) for p, _, _ in cis]
        df[f"{col}_95ci"] = [f"{lo:.0%}–{hi:.0%}" for _, lo, hi in cis]
    return df


def drift_tests(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per model: is turn 3 more often advice-like than turn 1 of the same conversation?"""
    rows = []
    for model, g in pairs.groupby("model"):
        before = (g["level_t1"] == 3).astype(int)
        after = (g["level_t3"] == 3).astype(int)
        test = mcnemar_exact(before, after)
        rows.append({
            "model": model,
            "conversations": len(g),
            "advice_like_turn1": f"{before.mean():.0%}",
            "advice_like_turn3": f"{after.mean():.0%}",
            "mean_level_change": round((g["level_t3"] - g["level_t1"]).mean(), 2),
            "became_advice": test["no_to_yes"],
            "stopped_being_advice": test["yes_to_no"],
            "mcnemar_p": round(test["p_value"], 4),
        })
    return pd.DataFrame(rows)


def agreement(con) -> pd.DataFrame:
    """Cohen's kappa between every pair of labellers (judges, people and the rule baseline)."""
    labels = con.execute("""
        SELECT response_id, 'judge: ' || label_source AS who, level, response FROM labelled_judge
        UNION ALL
        SELECT response_id, 'human: ' || label_source AS who, level, response FROM labelled_human
    """).df()
    if labels.empty:
        return pd.DataFrame()
    texts = labels.drop_duplicates("response_id").set_index("response_id")["response"]
    rule_labels = pd.DataFrame({
        "response_id": texts.index,
        "who": "rules baseline",
        "level": [detect_advice(t)["level"] for t in texts],
    })
    labels = pd.concat([labels[["response_id", "who", "level"]], rule_labels], ignore_index=True)
    wide = labels.pivot_table(index="response_id", columns="who", values="level", aggfunc="first")

    rows = []
    for a, b in combinations(wide.columns, 2):
        both = wide[[a, b]].dropna()
        if len(both) < 10 or not (a.startswith("human") or b.startswith("human")):
            continue  # only report agreement with at least one human, on 10+ shared answers
        kappa = cohen_kappa(both[a].astype(int), both[b].astype(int), labels=[0, 1, 2, 3])
        rows.append({
            "rater_a": a, "rater_b": b, "n": len(both),
            "exact_agreement": round((both[a] == both[b]).mean(), 3),
            "kappa": round(kappa, 3),
            "kappa_quadratic": round(cohen_kappa(both[a].astype(int), both[b].astype(int),
                                                 weights="quadratic", labels=[0, 1, 2, 3]), 3),
            "strength": kappa_label(kappa),
        })
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "_No rows yet._"
    df = df.head(max_rows)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def run_analysis(source: str = "judge", who: str | None = None, results_dir: Path = RESULTS,
        data_dir: Path = DATA, db_path: Path = DB_PATH) -> dict[str, pd.DataFrame]:
    con = build(results_dir=results_dir, data_dir=data_dir, db_path=db_path)
    use_labels(con, source, who)
    tables = {name: add_rates(con.execute(sql).df()) for name, sql in load_queries().items()}
    tables["drift_tests"] = drift_tests(tables["drift_pairs"]) if not tables["drift_pairs"].empty else pd.DataFrame()
    tables["agreement"] = agreement(con)

    out = Path(results_dir) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out / f"{source}_{name}.csv", index=False)

    note = ("> These use **LLM-judge labels**, which are silver labels until checked against human labels "
            "(see the agreement table)." if source == "judge" else "> These use **human labels**.")
    show = ["by_model", "by_detail", "by_ask_type", "over_refusal", "drift_by_turn", "drift_tests",
            "vulnerable_signposting", "agreement"]
    parts = [f"# AdviceLine results ({source} labels)", note]
    for name in show:
        df = tables[name]
        keep = [c for c in df.columns if not c.endswith("_rate")] if not df.empty else df.columns
        parts += [f"## {name.replace('_', ' ').capitalize()}", markdown_table(df[keep] if len(keep) else df)]
    summary = Path(results_dir) / f"summary_{source}.md"
    summary.write_text("\n\n".join(parts) + "\n")
    print(f"Wrote {len(tables)} tables to {out} and {summary}")
    return tables


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", choices=["judge", "human"], default="judge")
    p.add_argument("--who", default=None, help="one judge model or annotator name")
    p.add_argument("--results-dir", default=str(RESULTS))
    args = p.parse_args()
    run_analysis(args.labels, args.who, Path(args.results_dir))


if __name__ == "__main__":
    main()
