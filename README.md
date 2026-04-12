# Plant Disease Prediction and Remedy Assistant

This project uses a CNN model to detect plant diseases from leaf images and now also provides remedy guidance (local knowledge base + optional AI advice through OpenRouter).

## What is included

- Disease detection using TensorFlow/Keras
- Streamlit app for image upload and prediction
- Local remedy suggestions (immediate actions + prevention)
- Optional OpenRouter advice for richer treatment text
- Dataset downloader script for Kaggle dataset

## Project structure

```text
plant disease pridiction/
  dataset/
    train/
    valid/
  models/
    plant_disease_model.h5
    class_names.json
  src/
    __init__.py
    class_utils.py
    openrouter_integration.py
    predict.py
    preprocess.py
    remedy_knowledge.py
    train.py
  app.py
  config.py
  download_dataset.py
  requirements.txt
  .env
```

## Setup

1. Create virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Configure OpenRouter (optional, for AI advice):

Edit `.env`:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

## Dataset preparation

Option A: Use downloader script:

```powershell
python download_dataset.py
```

Option B: Manually place dataset folders here:

- `dataset/train/<class_name>/*.jpg`
- `dataset/valid/<class_name>/*.jpg`

## Train model

```powershell
python src/train.py
```

Outputs:
- `models/plant_disease_model.h5`
- `models/class_names.json`

## Run prediction from CLI

```powershell
python src/predict.py --image path_to_leaf.jpg
```

With AI advice:

```powershell
python src/predict.py --image path_to_leaf.jpg --ai-advice
```

## Run Streamlit app

```powershell
streamlit run app.py
```

Upload a leaf image and the app will show:
- Predicted disease class
- Confidence
- Local remedy steps
- Prevention recommendations
- Optional AI guidance

## Notes

- If you see `Model not found`, run training first.
- If AI advice fails, local remedy guidance still works.
- For real farm usage, verify final treatment with local agricultural extension experts.
