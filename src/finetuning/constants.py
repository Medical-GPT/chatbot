from pathlib import Path

INPUT_DIR = Path("../data/medical-conversations.txt")
MEDICAL_FINETUNING_FILE = Path("../data/data/medical-finetuning.txt")
EMPATHIC_FINETUNING_FILE = Path("../data/data/empathic-finetuning.txt")

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

FINETUNED_DIR = MODELS_DIR / "finetuned"
FINETUNED_DIR.mkdir(exist_ok=True)
