"""End to end on fake answers: generate -> judge -> DuckDB -> SQL -> statistics -> classifier."""

import pandas as pd

from adviceline import analysis, classify, generate, judge
from conftest import CannedBackend, RuleJudge


def make_results(project, prompts):
    for name in ["canned:model-a", "canned:model-b"]:
        generate.generate_answers(name, prompts, project["results"], backend=CannedBackend())
    judge.run_judge("rule-judge", project["results"], backend=RuleJudge())

    # Pretend a person labelled the first 40 answers, agreeing with the judge most of the time
    labels = pd.read_csv(next((project["results"] / "judge").glob("*.csv"))).head(40)
    labels["level"] = [lv if i % 5 else (lv + 1) % 4 for i, lv in enumerate(labels["level"])]
    labels["annotator"] = "tester"
    labels["notes"] = ""
    labels["labelled_at"] = "2026-10-06"
    (project["data"] / "labels").mkdir()
    cols = ["response_id", "annotator", "level", "refused", "signposted", "named_product", "notes", "labelled_at"]
    labels[cols].to_csv(project["data"] / "labels" / "tester.csv", index=False)


def test_full_pipeline(project, sample_prompts):
    make_results(project, sample_prompts)
    judged = pd.read_csv(next((project["results"] / "judge").glob("*.csv")))
    assert judged["parse_ok"].all()

    tables = analysis.run_analysis("judge", results_dir=project["results"], data_dir=project["data"], db_path=project["db"])
    by_model = tables["by_model"]
    assert set(by_model["model"]) == {"canned:model-a", "canned:model-b"}
    assert by_model["advice_like_rate"].between(0, 1).all()
    assert {"advice_like_95ci", "beyond_guidance_95ci"} <= set(by_model.columns)
    assert len(tables["drift_tests"]) == 2
    assert not tables["agreement"].empty
    assert (project["results"] / "summary_judge.md").exists()

    human = analysis.run_analysis("human", results_dir=project["results"], data_dir=project["data"], db_path=project["db"])
    assert human["by_model"]["n"].sum() <= 40

    df = classify.load_labelled("judge", results_dir=project["results"], data_dir=project["data"],
                                db_path=project["db"])
    report = classify.evaluate(df)
    assert 0 <= report["baseline"]["accuracy"] <= 1
    assert report["n_questions"] > 2
