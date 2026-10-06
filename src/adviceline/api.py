"""The guardrail as a web API.

Run:   uvicorn adviceline.api:app --reload
Docs:  http://127.0.0.1:8000/docs  (try the endpoints in your browser)
"""

from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from adviceline import __version__
from adviceline.guardrail import assess, guard, load_classifier

app = FastAPI(
    title="AdviceLine guardrail",
    version=__version__,
    description="Checks whether a chatbot's answer to a UK money question is advice-like, "
                "and rewrites it as guidance if it is. Research prototype, not legal advice.",
)
classifier = load_classifier()  # None until a model has been trained (python -m adviceline.classify)


class AnswerIn(BaseModel):
    question: str = Field("", description="The user's message")
    answer: str = Field(..., min_length=1, description="The chatbot's draft answer")
    topic: Optional[str] = Field(None, description="general, pensions, pension_transfer, debt or crypto")


class Assessment(BaseModel):
    level: int
    level_name: str
    refused: int
    signposted: int
    named_product: int
    flagged_sentences: List[str]
    method: str


class Guarded(BaseModel):
    original_level: int
    final_level: int
    changed: bool
    topic: str
    answer: str
    flagged_sentences: List[str]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "classifier_loaded": classifier is not None}


@app.post("/assess", response_model=Assessment)
def api_assess(body: AnswerIn) -> dict:
    return assess(body.answer, classifier)


@app.post("/guard", response_model=Guarded)
def api_guard(body: AnswerIn) -> dict:
    return guard(body.question, body.answer, topic=body.topic, classifier=classifier)
