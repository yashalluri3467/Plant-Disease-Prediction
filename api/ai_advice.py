from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Plant Disease AI Advice API")

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


@app.get("/")
def ai_advice_usage():
    return {
        "message": "Use POST /api/ai_advice with JSON body: {\"disease_name\": \"...\", \"crop_name\": \"...\"}"
    }


@app.post("/")
def ai_advice(payload: AdvicePayload):
    try:
        from src.openrouter_integration import get_disease_advice

        ai_response = get_disease_advice(payload.disease_name, payload.crop_name)
        return {"ai_advice": ai_response}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch AI advice: {exc}") from exc
