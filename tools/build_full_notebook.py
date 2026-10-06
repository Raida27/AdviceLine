"""Build notebooks/00_adviceline_all_code.ipynb: the whole project inside one Colab notebook.

Every project file becomes a `%%writefile` cell, so all the code is visible in the notebook and
running the cells rebuilds the same package as the repository. Re-run this script after changing
any file so the notebook never drifts from the code.

Run:  python tools/build_full_notebook.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "00_adviceline_all_code.ipynb"
RESULTS = "/content/drive/MyDrive/AdviceLine/results"

# (file, explanation shown above its cell), in the order you should read them
FILES = [
    ("pyproject.toml", "**Project settings.** The package name, the libraries it needs, and where the code lives (`src/`)."),
    ("data/questions.csv", "**The 60 questions:** 10 topics × (2 information + 2 guidance + 2 recommendation-seeking)."),
    ("data/personas.csv", "**One fictional persona per topic**, as a short version (a few details) and a full version."),
    ("data/vulnerabilities.csv", "**Six vulnerability statements** based on the FCA's vulnerability drivers (health, life events, capability)."),
    ("data/pressure.csv", "**Follow-up messages** that push for a direct recommendation in turns 2 and 3."),
    ("src/adviceline/__init__.py", "Marks `adviceline` as a Python package."),
    ("src/adviceline/config.py", "**Shared settings:** folders, the 4 rubric levels, flags and topic groups."),
    ("src/adviceline/prompts.py", "**Builds the 300 benchmark items** from the four CSV files: 60 questions × 4 detail levels, plus 60 three-turn conversations."),
    ("src/adviceline/backends.py", "**Talks to the language models.** Every backend has the same `chat()` method, so the rest of the code doesn't care whether a model runs on Colab's GPU, on Ollama or through an API."),
    ("src/adviceline/generate.py", "**Collects each model's answers**, saving after every batch so a disconnect never loses work."),
    ("src/adviceline/judge.py", "**The LLM judge.** A bigger model labels every answer with the rubric and returns JSON. These are *silver* labels until they're checked against human labels."),
    ("src/adviceline/rules.py", "**A rule-based detector** (regular expressions). It's the simple baseline the machine-learning models must beat, and the guardrail uses it to find advice-like sentences."),
    ("src/adviceline/metrics.py", "**The statistics:** Wilson confidence intervals, cluster bootstrap, Cohen's kappa, McNemar's test and calibration error."),
    ("src/adviceline/db.py", "**Loads everything into DuckDB** so the analysis can be written in SQL."),
    ("sql/analysis.sql", "**One SQL query per research question** (GROUP BY, FILTER, window functions)."),
    ("src/adviceline/analysis.py", "**Runs the SQL**, adds confidence intervals and the drift test, and writes a results summary."),
    ("src/adviceline/classify.py", "**Baseline classifier:** TF-IDF + logistic regression, evaluated with GroupKFold by question so there's no leakage."),
    ("src/adviceline/finetune.py", "**Fine-tunes ModernBERT** (a small transformer) to predict the level. Needs the GPU."),
    ("src/adviceline/guardrail.py", "**The guardrail:** detects advice-like answers and rewrites them as guidance, with signposting to free UK help."),
    ("src/adviceline/api.py", "**The guardrail as a web API** (FastAPI). Run it on your Mac with `uvicorn adviceline.api:app`."),
    ("src/adviceline/abtest.py", "**The blinded A/B test** of the guardrail: safety (McNemar) and helpfulness (Wilcoxon)."),
    ("app/streamlit_app.py", "**The Streamlit app:** results, an answer explorer, blind labelling and a guardrail demo. Run it on your Mac with `streamlit run app/streamlit_app.py`."),
    ("tests/conftest.py", "**Test helpers:** fake models and a fake judge, so the tests run without a GPU."),
    ("tests/test_prompts.py", "Tests for the benchmark design."),
    ("tests/test_metrics.py", "Tests for the statistics, checked against scikit-learn."),
    ("tests/test_rules.py", "Tests for the rule-based detector."),
    ("tests/test_judge.py", "Tests for reading the judge's JSON."),
    ("tests/test_generate.py", "Tests for collecting answers, including resuming after a stop."),
    ("tests/test_pipeline.py", "**End-to-end test:** generate → judge → DuckDB → SQL → statistics → classifier, on fake answers."),
    ("tests/test_guardrail_api.py", "Tests for the guardrail and the API."),
]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def writefile(path):
    content = (ROOT / path).read_text()
    return code(f"%%writefile {path}\n{content}")


def build():
    cells = [
        md("""
# AdviceLine: the complete project in one notebook
**When do AI chatbots cross the line from financial *guidance* into *advice*?**

All the code is in this notebook:
- **Part A** has one cell per file. Running a `%%writefile` cell saves that code as a file, so together they build a proper Python package.
- **Part B** runs the research on the GPU.

**Before you start:** Runtime → Change runtime type → **T4 GPU** → Save. Then run the cells from top to bottom.
Part A takes about a minute; Part B about 2 hours. Results are saved to Google Drive after every batch. If Colab disconnects, run everything again and it carries on where it stopped.

Labels follow a published rubric, not a legal ruling. All personas are fictional.
"""),
        md("## Part 0 · Setup"),
        code("# Is a GPU attached? You should see \"Tesla T4\"\n!nvidia-smi --query-gpu=name,memory.total --format=csv"),
        code(f"""
# Save results to Google Drive so a disconnect never loses work (click Allow in the pop-up)
import os
from google.colab import drive
drive.mount('/content/drive')
RESULTS = '{RESULTS}'
os.makedirs(RESULTS, exist_ok=True)
print('Results will be saved to', RESULTS)
"""),
        code("""
# Make the project folders and work inside them
import os
for folder in ['data', 'docs', 'sql', 'src/adviceline', 'app', 'tests', 'results']:
    os.makedirs(f'/content/adviceline/{folder}', exist_ok=True)
%cd /content/adviceline
"""),
        md("## Part A · The code\n### The rubric\nEvery label in the project follows these rules."),
        md((ROOT / "docs" / "rubric.md").read_text().replace("# AdviceLine labelling rubric", "#### AdviceLine labelling rubric", 1)),
        code("%%writefile docs/rubric.md\n" + (ROOT / "docs" / "rubric.md").read_text()),
    ]
    section, seen = None, set()
    titles = {"data": "### The benchmark data", "src": "### The Python package", "sql": "### The SQL",
              "app": "### The app (runs on your Mac)", "tests": "### Tests"}
    for path, explanation in FILES:
        top = path.split("/")[0]
        if top in titles and titles[top] != section:
            section = titles[top]
            cells.append(md(section + (" (continued)" if section in seen else "")))
            seen.add(section)
        cells.append(md(f"`{path}`: {explanation}"))
        cells.append(writefile(path))

    cells += [
        md("### Install the package and run the tests\nIf every test passes, the code above works."),
        code("!pip install -q -e \".[dev]\" bitsandbytes\n!python -m adviceline.prompts\n!python -m pytest -q"),
        md("## Part B · Run the research\n### B1. Build the 300 benchmark items"),
        code("""
!python -m adviceline.prompts
import pandas as pd
prompts = pd.read_csv('data/prompts.csv', keep_default_na=False)
prompts.sample(5, random_state=1)[['item_id', 'topic', 'ask_type', 'detail_level', 'turn1']]
"""),
        md("### B2. Quick test (about 3 minutes)\n5 single-turn items and 1 conversation through the first model, so problems show up early."),
        code("""
!python -m adviceline.generate --model hf:Qwen/Qwen3-4B-Instruct-2507 --limit 5 --results-dir /content/test_results
test = pd.read_csv('/content/test_results/responses/hf_Qwen_Qwen3-4B-Instruct-2507.csv')
for row in test.head(3).itertuples():
    print('USER:', row.user_message[:200]); print('MODEL:', row.response[:600]); print('-' * 80)
"""),
        md("""
### B3. Collect answers from the three models (about 15–20 minutes each)
| Model | Made by | Size | Licence |
|---|---|---|---|
| Qwen3-4B-Instruct-2507 | Alibaba | 4B | Apache 2.0 |
| Phi-4-mini-instruct | Microsoft | 3.8B | MIT |
| SmolLM3-3B | Hugging Face | 3B | Apache 2.0 |

Each model runs in its own process (`!python -m ...`) so the GPU memory is freed before the next one.
"""),
        code("""
MODELS = [
    'hf:Qwen/Qwen3-4B-Instruct-2507',
    'hf:microsoft/Phi-4-mini-instruct',
    'hf:HuggingFaceTB/SmolLM3-3B',
]
for m in MODELS:
    !python -m adviceline.generate --model {m} --results-dir {RESULTS}
"""),
        md("### B4. Label every answer with the LLM judge (about 30–45 minutes)\nQwen2.5-7B-Instruct, loaded in 4-bit to fit on the T4."),
        code("JUDGE = 'hf4bit:Qwen/Qwen2.5-7B-Instruct'\n!python -m adviceline.judge --judge {JUDGE} --results-dir {RESULTS}"),
        md("### B5. First look at the results (silver labels)"),
        code("""
!python -m adviceline.analysis --results-dir {RESULTS}
from IPython.display import Markdown, display
display(Markdown(open(f'{RESULTS}/summary_judge.md').read()))
"""),
        md("### B6. Guardrail A/B test: rewrite the advice-like answers, then label them again"),
        code("""
!python -m adviceline.guardrail --rewriter hf:Qwen/Qwen3-4B-Instruct-2507 --judge {JUDGE} --results-dir {RESULTS}
!python -m adviceline.judge --judge {JUDGE} --guarded --results-dir {RESULTS}
!python -m adviceline.abtest analyse --results-dir {RESULTS}
"""),
        md("### B7. Fine-tune the transformer classifier (about 10 minutes)"),
        code("!python -m adviceline.finetune --results-dir {RESULTS}"),
        md("### B8. Download the results\nThey also stay in Google Drive under **MyDrive/AdviceLine/results**."),
        code("""
!cd {RESULTS}/.. && zip -qr /content/adviceline_results.zip results -x 'results/models/*'
from google.colab import files
files.download('/content/adviceline_results.zip')
"""),
    ]
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
          "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"Wrote {OUT} ({len(cells)} cells, {sum(1 for c in cells if c['source'][:1] and c['source'][0].startswith('%%writefile'))} files)")


if __name__ == "__main__":
    build()
