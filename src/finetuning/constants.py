from pathlib import Path

INPUT_DIR = Path("../data/medical-conversations.txt")

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

FINETUNED_DIR = MODELS_DIR / "finetuned"
FINETUNED_DIR.mkdir(exist_ok=True)
