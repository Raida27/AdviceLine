"""Paths and constants shared by the whole project."""

from pathlib import Path

# The project root is two folders above this file: src/adviceline/config.py
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
DB_PATH = ROOT / "adviceline.duckdb"

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
