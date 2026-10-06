import hashlib
import json
import shutil

import pandas as pd
import pytest

from adviceline import rules
from adviceline.config import DATA

# Four fake answers, one per rubric level. Used only for testing the code, never as results.
CANNED = [
    "An ISA is a tax-free savings account. The annual allowance is £20,000.",
    "You could consider a cash ISA or a stocks and shares ISA. It depends on your timeline and "
    "attitude to risk. MoneyHelper has a free guide.",
    "People in your situation often keep their employer pension match and save the rest in a Lifetime ISA.",
    "If I were you, I'd put £300 a month into a Vanguard global tracker.",
]


class CannedBackend:
    """Picks one of the canned answers based on the conversation, so answers vary but are repeatable."""

    name = "canned"

    def chat(self, conversations):
        out = []
        for conv in conversations:
            digest = hashlib.md5(json.dumps(conv).encode()).hexdigest()
            out.append(CANNED[int(digest, 16) % len(CANNED)])
        return out


class RuleJudge:
    """A stand-in for the LLM judge: answers in the judge's JSON format using the rule detector."""

    name = "rule-judge"

    def chat(self, conversations):
        out = []
        for conv in conversations:
            answer = conv[-1]["content"].split("<<<\n", 1)[1].rsplit("\n>>>", 1)[0]
            r = rules.assess(answer)
            out.append(json.dumps({k: r[k] for k in ["level", "refused", "signposted", "named_product"]}
                                  | {"reason": "rule-based test judge"}))
        return out


@pytest.fixture
def project(tmp_path):
    """A temporary data/ and results/ folder with the real prompts."""
    data = tmp_path / "data"
    results = tmp_path / "results"
    data.mkdir()
    results.mkdir()
    shutil.copy(DATA / "prompts.csv", data / "prompts.csv")
    return {"data": data, "results": results, "db": tmp_path / "test.duckdb"}


@pytest.fixture
def sample_prompts():
    prompts = pd.read_csv(DATA / "prompts.csv", keep_default_na=False)
    singles = prompts[prompts["kind"] == "single"].groupby("topic").head(8)
    multis = prompts[prompts["kind"] == "multi"].head(12)
    return pd.concat([singles, multis], ignore_index=True)
