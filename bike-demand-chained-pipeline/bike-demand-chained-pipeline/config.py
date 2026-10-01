"""Central project configuration.

Keep experiment-wide constants here so the final notebook stays readable.
Replace TEAM_IDS with the registered team roster before running experiments.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"

TEAM_IDS: list[str] = []  # e.g. ["34521", "40218", "41190"]
VALIDATION_SIZE = 0.20
TARGET = "cnt"
