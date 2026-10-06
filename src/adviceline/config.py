"""Paths and constants shared by the whole project."""

import os
from pathlib import Path

try:
    # The project root is two folders above this file: src/adviceline/config.py
    ROOT = Path(__file__).resolve().parents[2]
except NameError:
    # Inside a notebook there is no __file__, so use the current folder
    ROOT = Path.cwd()
DATA = ROOT / "data"
# Set ADVICELINE_RESULTS to save results somewhere else (e.g. Google Drive in Colab)
RESULTS = Path(os.environ.get("ADVICELINE_RESULTS", ROOT / "results"))
DB_PATH = ROOT / "adviceline.duckdb"
BASELINE_MODEL_PATH = RESULTS / "models" / "baseline.joblib"

# The 4-level scale from docs/rubric.md
LEVELS = {
    0: "Information",
    1: "Guidance",
    2: "Targeted-support-like",
    3: "Personal recommendation",
}

# Extra yes/no flags from the rubric
FLAGS = ["refused", "signposted", "named_product"]

DETAIL_LEVELS = {
    0: "No personal details",
    1: "A few details",
    2: "Full persona",
    3: "Full persona + vulnerability",
}

ASK_TYPES = ["information", "guidance", "recommendation"]

# Pensions and investments are the only areas where the FCA's targeted support regime applies
TARGETED_SUPPORT_TOPICS = {
    "workplace_pensions",
    "retirement_income",
    "pension_transfers",
    "savings_isas",
    "investing",
}
