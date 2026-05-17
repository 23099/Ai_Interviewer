"""Project paths and shared configuration."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"
TRACKING_DIR = DATA_DIR / "tracking"
AUDIO_DIR = DATA_DIR / "audio"
QUESTION_BANK_PATH = DATA_DIR / "questions.json"
INTERVIEW_TEMPLATES_PATH = DATA_DIR / "interviews.json"

DEFAULT_ROLE_KEY = "software_engineer"

ROLE_LABELS = {
    "software_engineer": "Software Engineer",
    "data_analyst": "Data Analyst",
    "hr_assistant": "HR Assistant",
}

ACCOUNT_TYPES = {
    "hr": "HR",
    "employer": "Job Employer",
    "candidate": "Job Candidate",
}


def ensure_project_dirs() -> None:
    """Create runtime folders that should exist before saving interviews."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
