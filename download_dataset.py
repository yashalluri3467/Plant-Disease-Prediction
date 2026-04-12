import kagglehub
import shutil
from pathlib import Path

def download_and_organize_dataset():
    print("Downloading dataset from Kaggle...")

    # Download the dataset
    path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
    print(f"Path to dataset files: {path}")

    # Convert to Path object for easier handling
    dataset_path = Path(path)
    print(f"Dataset contents: {list(dataset_path.iterdir())}")

    # Define target directories
    base_dir = Path("dataset")
    train_dir = base_dir / "train"
    valid_dir = base_dir / "valid"

    # Create directories if they don't exist
    base_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)

    direct_train = dataset_path / "train"
    direct_valid = dataset_path / "valid"

    if direct_train.exists() and direct_valid.exists():
        source_train = direct_train
        source_valid = direct_valid
    else:
        source_train = next(dataset_path.rglob("train"), None)
        source_valid = next(dataset_path.rglob("valid"), None)

    # Check if the downloaded dataset already has train/valid structure
    if source_train and source_valid:
        print("Dataset already has train/valid structure. Copying files...")
        # Copy train directory
        if train_dir.exists():
            shutil.rmtree(train_dir)
        shutil.copytree(source_train, train_dir)

        # Copy valid directory
        if valid_dir.exists():
            shutil.rmtree(valid_dir)
        shutil.copytree(source_valid, valid_dir)
    else:
        print("Organizing dataset into train/valid split...")
        # Assume we need to create train/valid split from the downloaded data
        # For now, let's just copy everything to train and create a small validation set
        # In practice, you might want to split the data properly

        # Get all class directories
        class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]

        if not class_dirs:
            # If no subdirectories, the dataset might be flat
            print("Dataset appears to be flat. Please check the structure.")
            return

        for class_dir in class_dirs:
            class_name = class_dir.name
            print(f"Processing class: {class_name}")

            # Create class directories in train and valid
            (train_dir / class_name).mkdir(exist_ok=True)
            (valid_dir / class_name).mkdir(exist_ok=True)

            # Get all images in the class directory
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))

            if not images:
                print(f"No images found in {class_dir}")
                continue

            # Split 80% train, 20% valid
            split_idx = int(0.8 * len(images))
            train_images = images[:split_idx]
            valid_images = images[split_idx:]

            # Copy training images
            for img in train_images:
                shutil.copy2(img, train_dir / class_name / img.name)

            # Copy validation images
            for img in valid_images:
                shutil.copy2(img, valid_dir / class_name / img.name)

            print(f"  Copied {len(train_images)} training and {len(valid_images)} validation images")

    print(f"Dataset organized successfully!")
    print(f"Training data: {train_dir}")
    print(f"Validation data: {valid_dir}")

    # Print some statistics
    train_classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
    valid_classes = [d.name for d in valid_dir.iterdir() if d.is_dir()]
    print(f"Number of classes: {len(train_classes)}")
    print(f"Classes: {train_classes}")

if __name__ == "__main__":
    download_and_organize_dataset()
