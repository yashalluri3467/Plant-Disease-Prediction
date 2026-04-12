import argparse
import os

import tensorflow as tf
from tensorflow.keras import layers, models

from class_utils import save_class_names

try:
    from config import BATCH_SIZE, DATASET_DIR, EPOCHS, IMAGE_SIZE, MODEL_PATH
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from config import BATCH_SIZE, DATASET_DIR, EPOCHS, IMAGE_SIZE, MODEL_PATH


def build_datasets(batch_size: int):
    train_dir = os.path.join(DATASET_DIR, "train")
    valid_dir = os.path.join(DATASET_DIR, "valid")

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=True,
    )

    valid_dataset = tf.keras.utils.image_dataset_from_directory(
        valid_dir,
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=False,
    )

    class_names = train_dataset.class_names

    # Normalize once in the input pipeline.
    normalize = layers.Rescaling(1.0 / 255)
    autotune = tf.data.AUTOTUNE

    train_dataset = (
        train_dataset
        .map(lambda x, y: (normalize(x), y), num_parallel_calls=autotune)
        .prefetch(autotune)
    )
    valid_dataset = (
        valid_dataset
        .map(lambda x, y: (normalize(x), y), num_parallel_calls=autotune)
        .prefetch(autotune)
    )

    return train_dataset, valid_dataset, class_names


def build_model(num_classes: int):
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )

    model = models.Sequential(
        [
            layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
            data_augmentation,
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(epochs=None, steps_per_epoch=None, validation_steps=None):
    train_dataset, valid_dataset, class_names = build_datasets(BATCH_SIZE)
    model = build_model(num_classes=len(class_names))
    model.summary()

    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=epochs or EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
    )

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    saved_class_names = save_class_names({name: idx for idx, name in enumerate(class_names)})

    print(f"Model saved at {MODEL_PATH}")
    print(f"Class names saved ({len(saved_class_names)} classes).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="Limit training batches per epoch for faster runs",
    )
    parser.add_argument(
        "--validation-steps",
        type=int,
        default=None,
        help="Limit validation batches per epoch",
    )
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.validation_steps,
    )
