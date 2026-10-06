"""Load prompts, answers and labels into one DuckDB database so the analysis can be done in SQL.

Tables
  prompts        data/prompts.csv
  responses      results/responses/*.csv   (one file per model)
  judge_labels   results/judge/*.csv       (one file per judge model)
  human_labels   data/labels/*.csv         (one file per person labelling)

Views
  answers         responses joined to their prompt details
  labelled_judge  answers + judge labels
  labelled_human  answers + human labels

Run:  python -m adviceline.db
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from adviceline.config import DATA, DB_PATH, FLAGS, RESULTS

EMPTY = {
    "responses": ["response_id", "model", "item_id", "turn", "user_message", "response", "created_at"],
    "judge_labels": ["response_id", "judge", "parse_ok", "level", *FLAGS, "reason", "raw"],
    "human_labels": ["response_id", "annotator", "level", *FLAGS, "notes", "labelled_at"],
}


def _read_folder(folder: Path, table: str) -> pd.DataFrame:
    files = sorted(folder.glob("*.csv")) if folder.exists() else []
    if not files:
        return pd.DataFrame(columns=EMPTY[table])
    return pd.concat([pd.read_csv(f, keep_default_na=False) for f in files], ignore_index=True)


def _store(con: duckdb.DuckDBPyConnection, name: str, df: pd.DataFrame) -> None:
    con.register("incoming", df)
    con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM incoming")
    con.unregister("incoming")


def build(results_dir: Path = RESULTS, data_dir: Path = DATA, db_path: Path | str = DB_PATH):
    """(Re)build the database from the CSV files and return an open connection."""
    con = duckdb.connect(str(db_path))
    _store(con, "prompts", pd.read_csv(Path(data_dir) / "prompts.csv", keep_default_na=False))
    _store(con, "responses", _read_folder(Path(results_dir) / "responses", "responses"))
    _store(con, "judge_labels", _read_folder(Path(results_dir) / "judge", "judge_labels"))
    _store(con, "human_labels", _read_folder(Path(data_dir) / "labels", "human_labels"))

    flag_casts = ", ".join(f"TRY_CAST(l.{f} AS INTEGER) AS {f}" for f in FLAGS)
    con.execute("""
        CREATE OR REPLACE VIEW answers AS
        SELECT r.response_id, r.model, r.item_id, CAST(r.turn AS INTEGER) AS turn,
               r.user_message, r.response,
               p.kind, p.question_id, p.topic, p.ask_type,
               CAST(p.detail_level AS INTEGER) AS detail_level, p.vuln_id
        FROM responses r JOIN prompts p USING (item_id)
    """)
    con.execute(f"""
        CREATE OR REPLACE VIEW labelled_judge AS
        SELECT a.*, l.judge AS label_source, TRY_CAST(l.level AS INTEGER) AS level, {flag_casts}
        FROM answers a JOIN judge_labels l USING (response_id)
        WHERE TRY_CAST(l.parse_ok AS INTEGER) = 1
    """)
    con.execute(f"""
        CREATE OR REPLACE VIEW labelled_human AS
        SELECT a.*, l.annotator AS label_source, TRY_CAST(l.level AS INTEGER) AS level, {flag_casts}
        FROM answers a JOIN human_labels l USING (response_id)
    """)
    return con


def use_labels(con, source: str = "judge", who: str | None = None) -> None:
    """Point the view `labelled` (used by sql/analysis.sql) at one set of labels."""
    view = {"judge": "labelled_judge", "human": "labelled_human"}[source]
    where = ""
    if who:
        where = f"WHERE label_source = '{who.replace(chr(39), chr(39) * 2)}'"
    con.execute(f"CREATE OR REPLACE VIEW labelled AS SELECT * FROM {view} {where}")


def main() -> None:
    con = build()
    for table in ["prompts", "responses", "judge_labels", "human_labels"]:
        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"{table:>13}: {n} rows")
    print(f"Database: {DB_PATH}")


if __name__ == "__main__":
    main()
