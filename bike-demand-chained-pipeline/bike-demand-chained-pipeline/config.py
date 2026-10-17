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

TEAM_IDS: list[str] = ["16001061", "16008386", "16001322"]
EXPECTED_SEED = 36954  # sha256 of the sorted, underscore-joined IDs, mod 100000
VALIDATION_SIZE = 0.20
TARGET = "cnt"

PHASE1_DIR = ROOT / "experiments" / "phase1"
PHASE1_ARTIFACT = PHASE1_DIR / "phase1_artifact.json"
BONUS_DIR = ROOT / "experiments" / "phase1_bonus"
PHASE2_DIR = ROOT / "experiments" / "phase2"
PHASE2_ARTIFACT = PHASE2_DIR / "phase2_artifact.json"
PHASE3_DIR = ROOT / "experiments" / "phase3"
PHASE3_ARTIFACT = PHASE3_DIR / "phase3_artifact.json"
PHASE4_DIR = ROOT / "experiments" / "phase4"
PHASE4_ARTIFACT = PHASE4_DIR / "phase4_artifact.json"
PHASE5_DIR = ROOT / "experiments" / "phase5"
PHASE5_ARTIFACT = PHASE5_DIR / "phase5_artifact.json"
SUBMISSION_PATH = OUTPUT_DIR / "submission" / "sample_submission.csv"
