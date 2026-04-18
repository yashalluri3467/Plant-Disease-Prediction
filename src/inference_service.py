from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, Dict

from PIL import Image

from config import MODEL_PATH
from src.class_utils import load_class_names
from src.novel_class_handler import register_low_confidence_sample
from src.openrouter_integration import get_disease_advice, suggest_novel_class_from_image
from src.preprocess import preprocess_image
from src.remedy_knowledge import get_local_remedy

LOW_CONFIDENCE_THRESHOLD = 38.0


def get_top_predictions(predictions, class_names, top_k: int = 3):
    scores = [float(score) for score in predictions[0]]
    top_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]
    return [
        {
            "class": class_names[int(idx)],
            "confidence": float(scores[int(idx)] * 100),
        }
        for idx in top_indices
    ]


@lru_cache(maxsize=1)
def get_model():
    model_file = Path(MODEL_PATH)
    if not model_file.exists():
        raise RuntimeError(
            "Model file is not available in this deployment. "
            "Serverless deployments should call a dedicated ML inference service."
        )
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is not installed in this environment. "
            "Prediction endpoint is disabled to keep deployment under Vercel size limits."
        ) from exc
    return tf.keras.models.load_model(model_file.as_posix())


@lru_cache(maxsize=1)
def get_classes():
    return load_class_names()


def infer_from_bytes(image_bytes: bytes, source_filename: str, use_ai_advice: bool = False) -> Dict[str, Any]:
    image = Image.open(BytesIO(image_bytes))
    processed_image = preprocess_image(image)

    model = get_model()
    class_names = get_classes()
    predictions = model.predict(processed_image, verbose=0)
    scores = [float(score) for score in predictions[0]]
    predicted_index = max(range(len(scores)), key=lambda idx: scores[idx])
    confidence = float(scores[predicted_index] * 100)

    if predicted_index >= len(class_names):
        raise ValueError("Model output classes do not match class names. Re-train the model.")

    predicted_class = class_names[predicted_index]
    remedy = get_local_remedy(predicted_class)
    top_predictions = get_top_predictions(predictions, class_names, top_k=3)

    ai_advice = None
    if use_ai_advice:
        ai_advice = get_disease_advice(
            disease_name=remedy["disease"],
            crop_name=remedy["crop"],
        )

    low_confidence_entry = None
    llm_result = None
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        llm_result = suggest_novel_class_from_image(
            image_bytes=image_bytes,
            existing_class_count=len(class_names),
        )
        low_confidence_entry = register_low_confidence_sample(
            image_bytes=image_bytes,
            suggested_class_name=llm_result["suggested_class_name"],
            llm_description=llm_result["description"],
            source_filename=source_filename,
        )

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "top_predictions": top_predictions,
        "remedy": remedy,
        "ai_advice": ai_advice,
        "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
        "low_confidence_entry": low_confidence_entry,
        "backup_llm_result": llm_result,
    }
