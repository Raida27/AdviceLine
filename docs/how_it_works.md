# How AdviceLine works: a learning guide

You must be able to explain every file in an interview. Work through them in this order. For each one, **read it, run it, then rebuild the starred part from a blank file without looking**.

| # | File | What it teaches | Rebuild from scratch ⭐ | Interview question it answers |
|---|---|---|---|---|
| 1 | `docs/rubric.md` | Turning a fuzzy regulatory idea into labels people can apply consistently | Write the 4 levels in your own words | "How did you define advice-like?" |
| 2 | `src/adviceline/prompts.py` | pandas merges, f-strings, factorial experiment design | `build_items` | "Why 4 detail levels?" (to isolate the effect of personal detail) |
| 3 | `src/adviceline/backends.py` | Classes that share one method (an interface), HTTP requests, GPU generation | `DummyBackend` and `OllamaBackend` | "How did you compare local and hosted models fairly?" |
| 4 | `src/adviceline/generate.py` | Multi-turn conversations, saving checkpoints, resuming after a crash | `conversation()` | "What happens if Colab disconnects?" |
| 5 | `src/adviceline/judge.py` | LLM-as-a-judge, prompt design, robust JSON parsing | `parse_judgement` | "How do you know the judge is right?" (kappa vs human labels) |
| 6 | `src/adviceline/rules.py` | Regular expressions, a transparent baseline | Three of the patterns | "Why have a rule baseline?" (the ML has to beat something simple) |
| 7 | `src/adviceline/metrics.py` | Wilson intervals, Cohen's kappa, McNemar, calibration | `wilson_ci` and `cohen_kappa` | "Why Wilson rather than p ± 1.96·SE?" |
| 8 | `src/adviceline/db.py` + `sql/analysis.sql` | DuckDB, joins, views, GROUP BY, FILTER, window functions (LAG) | `by_model` and `drift_pairs` | "Show me a window function you've used." |
| 9 | `src/adviceline/analysis.py` | Putting SQL and statistics together into tables | `add_rates` | "How did you test for drift?" (paired McNemar) |
| 10 | `src/adviceline/classify.py` | TF-IDF, logistic regression, GroupKFold, class imbalance | `make_pipeline` + `evaluate` | "How did you avoid data leakage?" (group by question) |
| 11 | `src/adviceline/finetune.py` | A PyTorch training loop, mixed precision, class weights | The training loop | "Why fine-tune if TF-IDF works?" |
| 12 | `src/adviceline/guardrail.py` + `api.py` | FastAPI, Pydantic validation, safe defaults | The `/guard` endpoint | "How would this run in production?" |
| 13 | `src/adviceline/abtest.py` | Blinded A/B design, paired tests | `make_sheet` | "How did you measure the cost of the guardrail?" |
| 14 | `app/streamlit_app.py` | Dashboards and annotation tools | The guardrail page | "Who would use this and how?" |
| 15 | `tests/` | pytest, fixtures, end-to-end tests with fake data | `test_metrics.py` | "How do you know your code works?" |

## Your parts (these must be your own work)
1. **Human labels:** label 300 answers in the app, and ask a friend to label the first 100. This is what makes the research credible.
2. **Reading the results:** write what you found in plain English, including what surprised you.
3. **The paper and the policy brief:** your words, your argument.

## The 2-minute project story (STAR)
- **Situation:** In 2026 the FCA opened a new middle ground between guidance and advice, and its Mills Review warned that chatbots may already give advice-like answers outside the rules.
- **Task:** Measure how often that happens, and build something that fixes it.
- **Action:** I built a 300-item UK benchmark, ran 3 open models on a GPU, labelled answers with an LLM judge checked against my own labels, trained classifiers, and built a guardrail API tested by a blinded A/B experiment.
- **Result:** *(fill in with your numbers: advice-like rate, drift, kappa, guardrail effect and helpfulness cost)*
