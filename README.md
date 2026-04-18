# Plant Disease Prediction Web App (React + Python API)

This project is now a web-first app:

- Frontend: React (Vite), HTML, CSS
- Backend API: FastAPI (prediction + optional OpenRouter advice)
- Deployment target: Vercel

## Project structure

```text
plant disease pridiction/
  api/
    _inference.py
    predict.py
    ai_advice.py
    health.py
  frontend/
    src/
      App.jsx
      api.js
      main.jsx
      styles.css
    index.html
    package.json
    vite.config.js
  models/
    plant_disease_model.h5
    class_names.json
  src/
    class_utils.py
    novel_class_handler.py
    openrouter_integration.py
    predict.py
    preprocess.py
    remedy_knowledge.py
    train.py
  config.py
  requirements.txt
  vercel.json
```

## Local development

### 1) Python setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Frontend setup

```powershell
cd frontend
npm install
npm run dev
```

Frontend starts on `http://localhost:5173`.

### 3) Run API locally

In another terminal:

```powershell
.\.venv\Scripts\activate
uvicorn api.predict:app --reload --port 8001
```

Optional AI advice endpoint:

```powershell
uvicorn api.ai_advice:app --reload --port 8002
```

Set frontend API base URL:

```powershell
$env:VITE_API_BASE_URL='http://localhost:8001'
```

For AI advice via separate port, you can proxy or keep `use_ai_advice` enabled in `/api/predict`.

## OpenRouter setup (optional)

Create `.env` file:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_VISION_BACKUP_MODEL=openai/gpt-4o-mini
```

If no valid key is present, local remedy logic still works and AI responses degrade gracefully.

## Train / update model

```powershell
python src/train.py
```

Outputs:

- `models/plant_disease_model.h5`
- `models/class_names.json`

## Vercel deployment

### 1) Push project to GitHub

### 2) Import repository in Vercel

Vercel uses:

- `buildCommand`: `npm run build`
- `outputDirectory`: `frontend/dist`
- Python API functions from `api/*.py`

### 3) Set environment variables in Vercel

- `OPENROUTER_API_KEY` (optional)
- `OPENROUTER_VISION_BACKUP_MODEL` (optional)

### 4) Deploy

Frontend serves from static build, and API endpoints are available as:

- `/api/predict`
- `/api/ai_advice`
- `/api/health`

## Important deployment note

`models/plant_disease_model.h5` is large (~533 MB) and cannot fit in Vercel serverless limits.

- Vercel deployment should be used for frontend + lightweight API routes (like `/api/ai_advice`).
- Run full TensorFlow model inference (`/api/predict`) on a dedicated backend service and connect it from the frontend.
