import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "plant_disease_model.h5")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

OPENROUTER_MODEL = "anthropic/claude-3-haiku"
OPENROUTER_TIMEOUT_SECONDS = 30
