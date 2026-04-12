import argparse
from functools import lru_cache

import numpy as np
import tensorflow as tf

try:
    from .class_utils import load_class_names
    from .openrouter_integration import get_disease_advice
    from .preprocess import preprocess_image
    from .remedy_knowledge import get_local_remedy
except ImportError:
    from class_utils import load_class_names
    from openrouter_integration import get_disease_advice
    from preprocess import preprocess_image
    from remedy_knowledge import get_local_remedy

try:
    from config import MODEL_PATH
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from config import MODEL_PATH


@lru_cache(maxsize=1)
def load_model_cached():
    return tf.keras.models.load_model(MODEL_PATH)


def get_top_predictions(predictions: np.ndarray, class_names, top_k: int = 3):
    scores = np.squeeze(predictions)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        {
            "class": class_names[int(idx)],
            "confidence": float(scores[int(idx)] * 100),
        }
        for idx in top_indices
    ]


def predict(image_path: str, use_ai_advice: bool = False):
    model = load_model_cached()
    image = preprocess_image(image_path)
    predictions = model.predict(image, verbose=0)

    class_names = load_class_names()
    predicted_idx = int(np.argmax(predictions))

    if predicted_idx >= len(class_names):
        raise ValueError("Model output classes do not match available class names.")

    predicted_class = class_names[predicted_idx]
    confidence = float(np.max(predictions) * 100)
    local_remedy = get_local_remedy(predicted_class)

    ai_advice = None
    if use_ai_advice:
        ai_advice = get_disease_advice(local_remedy["disease"], crop_name=local_remedy["crop"])

    return predicted_class, confidence, local_remedy, ai_advice


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--ai-advice",
        action="store_true",
        help="Fetch additional remedy guidance from OpenRouter",
    )
    args = parser.parse_args()

    label, confidence, local_remedy, ai_advice = predict(args.image, use_ai_advice=args.ai_advice)
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2f}%")

    print("\nLocal Remedy:")
    print(local_remedy["summary"])
    print("Actions:")
    for action in local_remedy["actions"]:
        print(f"- {action}")
    print("Prevention:")
    for item in local_remedy["prevention"]:
        print(f"- {item}")
    print(f"Safety Note: {local_remedy['safety_note']}")

    if ai_advice:
        print("\nAI Advice:")
        print(ai_advice)
