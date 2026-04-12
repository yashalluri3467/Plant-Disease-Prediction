import json
import os
from typing import Dict, List

try:
    from config import CLASS_NAMES_PATH, DATASET_DIR
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from config import CLASS_NAMES_PATH, DATASET_DIR


def save_class_names(class_indices: Dict[str, int]) -> List[str]:
    ordered_names = [name for name, _ in sorted(class_indices.items(), key=lambda item: item[1])]
    os.makedirs(os.path.dirname(CLASS_NAMES_PATH), exist_ok=True)

    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as fp:
        json.dump(ordered_names, fp, indent=2)

    return ordered_names


def load_class_names() -> List[str]:
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as fp:
            return json.load(fp)

    train_dir = os.path.join(DATASET_DIR, "train")
    if os.path.isdir(train_dir):
        class_names = sorted(
            entry for entry in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, entry))
        )
        if class_names:
            return class_names

    raise FileNotFoundError(
        "Class names not found. Train the model first or place class folders in dataset/train."
    )
