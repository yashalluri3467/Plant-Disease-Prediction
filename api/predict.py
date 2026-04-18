from pydantic import BaseModel
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Plant Disease Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AdvicePayload(BaseModel):
    disease_name: str
    crop_name: str | None = None


@app.post("/")
@app.post("/api/predict")
async def predict(file: UploadFile = File(...), use_ai_advice: str = Form("false")):
    try:
        from src.inference_service import infer_from_bytes

        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        normalized_advice_flag = use_ai_advice.strip().lower()
        parsed_advice_flag = normalized_advice_flag in {"1", "true", "yes", "on"}

        result = infer_from_bytes(
            image_bytes=image_bytes,
            source_filename=file.filename or "uploaded_image.jpg",
            use_ai_advice=parsed_advice_flag,
        )
        return result
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.post("/api/ai_advice")
def ai_advice(payload: AdvicePayload):
    try:
        from src.openrouter_integration import get_disease_advice

        ai_response = get_disease_advice(payload.disease_name, payload.crop_name)
        return {"ai_advice": ai_response}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch AI advice: {exc}") from exc
