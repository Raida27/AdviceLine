import pandas as pd

from adviceline.backends import DummyBackend, slug
from adviceline.generate import conversation, run


def test_slug_is_filename_safe():
    assert slug("hf:Qwen/Qwen3-4B-Instruct-2507") == "hf_Qwen_Qwen3-4B-Instruct-2507"


def test_multi_turn_conversation_includes_earlier_answers():
    item = {"item_id": "M-x", "turn1": "q1", "turn2": "q2", "turn3": "q3"}
    previous = {("M-x", 1): "a1", ("M-x", 2): "a2"}
    msgs = conversation(item, 3, previous)
    assert [m["content"] for m in msgs] == ["q1", "a1", "q2", "a2", "q3"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant", "user"]


def test_run_writes_one_row_per_turn_and_resumes(project, sample_prompts):
    path = run("dummy", sample_prompts, project["results"], chunk=7, backend=DummyBackend())
    df = pd.read_csv(path)
    n_single = (sample_prompts["kind"] == "single").sum()
    n_multi = (sample_prompts["kind"] == "multi").sum()
    assert len(df) == n_single + 3 * n_multi
    assert df["response_id"].is_unique
    assert set(df[df["item_id"].str.startswith("M-")]["turn"]) == {1, 2, 3}

    run("dummy", sample_prompts, project["results"], backend=DummyBackend())  # nothing left to do
    assert len(pd.read_csv(path)) == len(df)
