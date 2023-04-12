from pathlib import Path

INPUT_DIR = Path("../data/data/pretraining/preprocessed.txt")

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

PRETRAINED_DIR = MODELS_DIR / "pretrained"
PRETRAINED_DIR.mkdir(exist_ok=True)

CHECKPOINT_DIR = PRETRAINED_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

ENCODER_DIR = Path("encoder")
ENCODER_DIR.mkdir(exist_ok=True)
ENCODER_TOKENS = ENCODER_DIR / "tokens.txt"
ENCODER_ENCTEXT = ENCODER_DIR / "encoded.bin"
