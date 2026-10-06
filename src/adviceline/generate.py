"""Collect a model's answers to every benchmark item.

Single-turn items get one answer. Three-turn items get three: after each answer the next
user message is added, and the model answers again with the whole conversation so far.

Answers are saved to results/responses/<model>.csv, one row per (item, turn), after every
chunk. Finished rows are skipped, so after a Colab disconnect just run the same command again.

Run:
    python -m adviceline.generate --model hf:Qwen/Qwen3-4B-Instruct-2507
    python -m adviceline.generate --model dummy --limit 10        # quick test, no GPU
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from adviceline.backends import make_backend, slug
from adviceline.config import DATA, RESULTS
from adviceline.prompts import turns

COLUMNS = ["response_id", "model", "item_id", "turn", "user_message", "response", "created_at"]


def load_done(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, keep_default_na=False)
    return pd.DataFrame(columns=COLUMNS)


def conversation(item: dict, turn: int, previous: dict) -> list[dict]:
    """The messages the model sees when answering `turn` of `item`."""
    user_msgs = turns(item)
    messages = []
    for t in range(1, turn + 1):
        messages.append({"role": "user", "content": user_msgs[t - 1]})
        if t < turn:
            messages.append({"role": "assistant", "content": previous[(item["item_id"], t)]})
    return messages


def generate_answers(model_spec: str, prompts: pd.DataFrame, results_dir: Path = RESULTS,
        chunk: int = 64, backend=None) -> Path:
    out_path = Path(results_dir) / "responses" / f"{slug(model_spec)}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = load_done(out_path)
    answers = {(r["item_id"], int(r["turn"])): r["response"] for r in done.to_dict("records")}
    items = prompts.fillna("").to_dict("records")
    backend = backend or make_backend(model_spec)
    started = time.time()

    for turn in (1, 2, 3):
        todo = [it for it in items
                if len(turns(it)) >= turn
                and (it["item_id"], turn) not in answers
                and (turn == 1 or (it["item_id"], turn - 1) in answers)]
        if not todo:
            continue
        print(f"[{model_spec}] turn {turn}: {len(todo)} answers to write", flush=True)
        for start in range(0, len(todo), chunk):
            batch = todo[start:start + chunk]
            convs = [conversation(it, turn, answers) for it in batch]
            replies = backend.chat(convs)
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            rows = []
            for it, conv, reply in zip(batch, convs, replies):
                answers[(it["item_id"], turn)] = reply
                rows.append({
                    "response_id": f"{slug(model_spec)}|{it['item_id']}|t{turn}",
                    "model": model_spec,
                    "item_id": it["item_id"],
                    "turn": turn,
                    "user_message": conv[-1]["content"],
                    "response": reply,
                    "created_at": now,
                })
            # Save after every chunk so nothing is lost if the session stops
            pd.DataFrame(rows, columns=COLUMNS).to_csv(
                out_path, mode="a", header=not out_path.exists(), index=False)
            minutes = (time.time() - started) / 60
            print(f"  saved {start + len(batch)}/{len(todo)}  ({minutes:.1f} min)", flush=True)

    print(f"Done: {out_path}")
    return out_path


def quick_sample(prompts: pd.DataFrame, n: int) -> pd.DataFrame:
    """The first n single-turn items plus a few conversations, for a quick test."""
    return pd.concat([prompts[prompts["kind"] == "single"].head(n),
                      prompts[prompts["kind"] == "multi"].head(max(1, n // 5))])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="backend spec, e.g. hf:Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--results-dir", default=str(RESULTS))
    p.add_argument("--limit", type=int, default=None, help="only the first N items (for a quick test)")
    p.add_argument("--chunk", type=int, default=64, help="answers per save")
    args = p.parse_args()

    prompts = pd.read_csv(DATA / "prompts.csv", keep_default_na=False)
    if args.limit:
        prompts = quick_sample(prompts, args.limit)
    generate_answers(args.model, prompts, Path(args.results_dir), chunk=args.chunk)


if __name__ == "__main__":
    main()
