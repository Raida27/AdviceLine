from fastapi.testclient import TestClient

from adviceline.api import app
from adviceline.guardrail import detect_topic, guard, rewrite_without_llm

client = TestClient(app)
ADVICE = "Great question! If I were you, I'd open a Lifetime ISA. A Lifetime ISA adds a 25% bonus."


def test_topic_detection():
    assert detect_topic("Should I transfer my final salary pension?") == "pension_transfer"
    assert detect_topic("Which credit card should I pay off first?") == "debt"
    assert detect_topic("Should I buy Bitcoin?") == "crypto"
    assert detect_topic("Should I use a cash ISA?") == "general"


def test_rewrite_removes_advice_and_adds_signposting():
    new = rewrite_without_llm(ADVICE, "general")
    assert "If I were you" not in new
    assert "25% bonus" in new
    assert "moneyhelper.org.uk" in new


def test_guard_uses_llm_backend_when_given():
    class FakeLLM:
        def chat(self, conversations):
            assert "Do NOT tell the person what to do" in conversations[0][0]["content"]
            return ["A Lifetime ISA adds a 25% bonus. Options depend on your plans."]

    result = guard("ISA?", ADVICE, backend=FakeLLM())
    assert result["changed"] and result["original_level"] == 3 and result["final_level"] < 2


def test_api_health():
    assert client.get("/health").json()["status"] == "ok"


def test_api_assess_flags_advice():
    body = client.post("/assess", json={"answer": ADVICE}).json()
    assert body["level"] == 3 and body["level_name"] == "Personal recommendation"


def test_api_guard_rewrites_advice_but_not_information():
    body = client.post("/guard", json={"question": "Should I open a Lifetime ISA?", "answer": ADVICE}).json()
    assert body["changed"] and body["final_level"] < 2
    info = "The Lifetime ISA bonus is 25%, up to £1,000 a year."
    body = client.post("/guard", json={"answer": info}).json()
    assert not body["changed"] and body["answer"] == info


def test_api_rejects_empty_answer():
    assert client.post("/guard", json={"answer": ""}).status_code == 422
