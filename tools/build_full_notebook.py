"""Build notebooks/AdviceLine.ipynb: the whole project as plain, runnable Python cells.

Each module becomes one code cell that runs directly in the notebook (no files, no GitHub, no
`!python -m`). The command-line parts (`main()`, `if __name__ == "__main__"`) and the imports
between modules are removed, because in a notebook every cell shares one namespace. The data
files become cells too. Results are saved to Google Drive.

Re-run this script after changing any file so the notebook never drifts from the package.

Run:  python tools/build_full_notebook.py
"""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "AdviceLine.ipynb"
DRIVE_RESULTS = "/content/drive/MyDrive/AdviceLine/results"

DATA_FILES = [
    ("questions.csv", "The 60 questions: 10 topics × (2 information + 2 guidance + 2 recommendation-seeking)."),
    ("personas.csv", "One fictional persona per topic, as a short version (a few details) and a full version."),
    ("vulnerabilities.csv", "Six vulnerability statements based on the FCA's vulnerability drivers."),
    ("pressure.csv", "Follow-up messages that push for a direct recommendation in turns 2 and 3."),
]

# (module, one-line summary). The order matters: each cell may use names from the cells above it.
MODULES = [
    ("config", "Shared settings: folders, the 4 rubric levels, flags and topic groups."),
    ("prompts", "Builds the 300 benchmark items from the four data files."),
    ("backends", "Talks to the language models. Every backend has the same `chat()` method."),
    ("generate", "Collects each model's answers, saving after every batch."),
    ("judge", "The LLM judge: labels every answer with the rubric."),
    ("rules", "A rule-based detector: the simple baseline, also used by the guardrail."),
    ("metrics", "The statistics: confidence intervals, kappa, McNemar, calibration."),
    ("db", "Loads everything into DuckDB so the analysis can be written in SQL."),
    ("SQL", "One SQL query per research question."),
    ("analysis", "Runs the SQL, adds confidence intervals and the drift test, writes a summary."),
    ("classify", "Baseline classifier: TF-IDF + logistic regression, GroupKFold by question."),
    ("finetune", "Fine-tunes a small transformer (ModernBERT) on the GPU."),
    ("guardrail", "Detects advice-like answers and rewrites them as guidance, with signposting."),
    ("api", "The guardrail as a web API (FastAPI)."),
    ("abtest", "The blinded A/B test of the guardrail."),
]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text, library=False):
    meta = {"tags": ["library"]} if library else {}
    return {"cell_type": "code", "execution_count": None, "metadata": meta, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def explain(docstring):
    """The module docstring without its command-line instructions."""
    keep = []
    for line in (docstring or "").splitlines():
        if line.strip().startswith(("Run", "Batch use", "Docs:")):
            break
        keep.append(line)
    return "\n".join(keep).strip()


def notebook_code(module):
    """A module's source, minus its docstring, inter-module imports and command-line entry point."""
    source = (ROOT / "src" / "adviceline" / f"{module}.py").read_text()
    tree = ast.parse(source)
    drop = set()
    for node in tree.body:
        is_docstring = (node is tree.body[0] and isinstance(node, ast.Expr)
                        and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str))
        is_future = isinstance(node, ast.ImportFrom) and node.module == "__future__"
        is_internal = (isinstance(node, ast.ImportFrom) and (node.module or "").startswith("adviceline")) or \
                      (isinstance(node, ast.Import) and any(a.name.startswith("adviceline") for a in node.names))
        is_main = isinstance(node, ast.FunctionDef) and node.name == "main"
        is_entry = isinstance(node, ast.If) and "__name__" in ast.unparse(node.test)
        if is_docstring or is_future or is_internal or is_main or is_entry:
            drop.update(range(node.lineno, node.end_lineno + 1))
    lines = [line for i, line in enumerate(source.splitlines(), start=1) if i not in drop]
    text = "\n".join(lines).strip("\n")
    while "\n\n\n\n" in text:
        text = text.replace("\n\n\n\n", "\n\n\n")
    if "argparse." not in text:
        text = text.replace("import argparse\n", "")
    return ast.get_docstring(tree), text


def top_level_names(text):
    names = set()
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def build():
    cells = [md(f"""
# AdviceLine ⚖️
**When do AI chatbots cross the line from financial *guidance* into *advice*?** A UK benchmark, a classifier and a guardrail.

**Every cell is plain Python that runs right here.**
- **Part A** sets up the project: the data and every function. It takes seconds and needs no GPU.
- **Part B** runs the research on the GPU. Every result is saved to Google Drive in `{DRIVE_RESULTS.replace('/content/drive/', '')}`.

**Before you start:** Runtime → Change runtime type → **T4 GPU** → Save. Then run the cells from top to bottom (or Runtime → Run all).
If Colab disconnects, reconnect and run all the cells again. Finished work is skipped, so it carries on where it stopped.

*Research, not legal advice: labels follow a published rubric (below), not a legal ruling. All personas are fictional.*
"""),
        md("## Part 0 · Setup"),
        code("# Is a GPU attached? You should see \"Tesla T4\"\n!nvidia-smi --query-gpu=name,memory.total --format=csv"),
        code("# Libraries Colab doesn't include by default\n!pip install -q duckdb bitsandbytes fastapi httpx"),
        code(f"""
import os
from google.colab import drive

drive.mount('/content/drive')  # click Allow in the pop-up

# Results (answers, labels, tables) go to Google Drive, so nothing is lost if Colab disconnects
os.environ['ADVICELINE_RESULTS'] = '{DRIVE_RESULTS}'

# A working folder for the data files and the database
for folder in ['/content/adviceline/data', '/content/adviceline/sql']:
    os.makedirs(folder, exist_ok=True)
os.chdir('/content/adviceline')
print('Working folder:', os.getcwd())
print('Results folder:', os.environ['ADVICELINE_RESULTS'])
"""),
        md("## Part A · The project\n### The rubric\nEvery label in the project follows these rules."),
        md((ROOT / "docs" / "rubric.md").read_text().replace("# AdviceLine labelling rubric", "#### AdviceLine labelling rubric", 1)),
        md("### The benchmark data\nThe four hand-written tables that the 300 items are built from."),
    ]
    for name, about in DATA_FILES:
        content = (ROOT / "data" / name).read_text()
        assert '"""' not in content and "\\" not in content, name
        cells.append(code(f'# {about}\nfrom pathlib import Path\n\nPath("data/{name}").write_text("""{content}""")', library=True))

    cells.append(md("### The code\nEach cell below is one part of the project. Run them in order: later cells use functions from earlier ones."))
    init = (ROOT / "src" / "adviceline" / "__init__.py").read_text()
    cells.append(code(init.split('"""', 2)[2].strip(), library=True))  # __version__

    seen = {}
    for number, (module, summary) in enumerate(MODULES, start=1):
        if module == "SQL":
            sql = (ROOT / "sql" / "analysis.sql").read_text()
            assert '"""' not in sql
            cells.append(md(f"#### {number}. SQL\n**{summary}** Every query reads the view `labelled` (answers + one set of labels)."))
            cells.append(code(f'ANALYSIS_SQL = """{sql}"""\n\nPath("sql/analysis.sql").write_text(ANALYSIS_SQL)', library=True))
            continue
        doc, text = notebook_code(module)
        for n in top_level_names(text):
            if n in seen:
                raise SystemExit(f"Name clash: {n} is defined in both {seen[n]} and {module}")
            seen[n] = module
        cells.append(md(f"#### {number}. `{module}`\n**{summary}**\n\n{explain(doc)}"))
        cells.append(code(text, library=True))

    cells += [
        md("## Part B · Run the research\n### B1. Build the 300 benchmark items"),
        code("""
items = write_prompts()
items.sample(5, random_state=1)[['item_id', 'topic', 'ask_type', 'detail_level', 'turn1']]
"""),
        md("### B2. Quick test (about 3 minutes)\n5 single-turn items and 1 conversation through the first model, so problems show up early."),
        code("""
from pathlib import Path
import pandas as pd

backend = make_backend('hf:Qwen/Qwen3-4B-Instruct-2507')  # downloads the model the first time
test_file = generate_answers('hf:Qwen/Qwen3-4B-Instruct-2507', quick_sample(items, 5),
                             Path('/content/test_results'), backend=backend)
for row in pd.read_csv(test_file).head(3).itertuples():
    print('USER:', row.user_message[:200])
    print('MODEL:', row.response[:600])
    print('-' * 80)
"""),
        md("""
### B3. Collect answers from the three models (about 15–20 minutes each)
| Model | Made by | Size | Licence |
|---|---|---|---|
| Qwen3-4B-Instruct-2507 | Alibaba | 4B | Apache 2.0 |
| Phi-4-mini-instruct | Microsoft | 3.8B | MIT |
| SmolLM3-3B | Hugging Face | 3B | Apache 2.0 |
"""),
        code("""
MODELS = [
    'hf:Qwen/Qwen3-4B-Instruct-2507',
    'hf:microsoft/Phi-4-mini-instruct',
    'hf:HuggingFaceTB/SmolLM3-3B',
]
for spec in MODELS:
    backend = None
    free_gpu()  # release the previous model's memory first
    backend = make_backend(spec)
    generate_answers(spec, items, RESULTS, backend=backend)
backend = None
free_gpu()
"""),
        md("### B4. Label every answer with the LLM judge (about 30–45 minutes)\nQwen2.5-7B-Instruct, loaded in 4-bit to fit on the T4. These are *silver* labels until you check them against your own."),
        code("""
JUDGE = 'hf4bit:Qwen/Qwen2.5-7B-Instruct'
run_judge(JUDGE, RESULTS)
free_gpu()
"""),
        md("### B5. First look at the results"),
        code("""
from IPython.display import Markdown, display

tables = run_analysis('judge', results_dir=RESULTS)
display(Markdown((RESULTS / 'summary_judge.md').read_text()))
"""),
        md("### B6. Guardrail A/B test: rewrite the advice-like answers, then label them again"),
        code("""
rewrite_batch('hf:Qwen/Qwen3-4B-Instruct-2507', RESULTS, judge=JUDGE)
free_gpu()
run_judge(JUDGE, RESULTS, guarded=True)
free_gpu()
safety(RESULTS)
"""),
        md("### B7. Baseline classifier (no GPU needed)\nCompare its kappa with the rule-based detector's."),
        code("run_classifier('judge', results_dir=RESULTS)"),
        md("### B8. Fine-tune the transformer classifier (about 10 minutes)"),
        code("finetune_classifier(results_dir=RESULTS)"),
        md("### B9. Try the guardrail API"),
        code("""
from fastapi.testclient import TestClient

client = TestClient(app)
draft = "If I were you, I'd open a Lifetime ISA and put £4,000 a year into it."
client.post('/guard', json={'question': 'Should I save in a Lifetime ISA?', 'answer': draft}).json()
"""),
        md("### B10. Download the results (optional)\nThey also stay in Google Drive."),
        code("""
!cd {RESULTS}/.. && zip -qr /content/adviceline_results.zip results -x 'results/models/*'
from google.colab import files
files.download('/content/adviceline_results.zip')
"""),
        md("## Appendix · Code that runs outside Colab\nShown here so you can read it all in one place. These don't run in Colab: the app runs on your Mac (`streamlit run app/streamlit_app.py`) and the tests run with `pytest` on your Mac and on GitHub."),
        md("#### The Streamlit app (`app/streamlit_app.py`)\n```python\n" + (ROOT / "app" / "streamlit_app.py").read_text() + "```"),
    ]
    for test in sorted((ROOT / "tests").glob("*.py")):
        cells.append(md(f"#### `tests/{test.name}`\n```python\n{test.read_text()}```"))

    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
          "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    n_code = sum(c["cell_type"] == "code" for c in cells)
    print(f"Wrote {OUT}: {len(cells)} cells ({n_code} code), no name clashes across {len(seen)} names")


if __name__ == "__main__":
    build()
