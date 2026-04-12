from typing import Any

import numpy as np
from PIL import Image

try:
    from config import IMAGE_SIZE
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from config import IMAGE_SIZE


def preprocess_image(image_source: Any):
    """Preprocess image source into model-ready tensor.

    Supports file path, file-like object (e.g. Streamlit UploadedFile), or PIL.Image.
    """
    if isinstance(image_source, Image.Image):
        img = image_source
    else:
        if hasattr(image_source, "seek"):
            image_source.seek(0)
        img = Image.open(image_source)

    with img:
        rgb_image = img.convert("RGB")
        resized_image = rgb_image.resize(IMAGE_SIZE)
        image = np.array(resized_image, dtype=np.float32)

    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image
