"""FastAPI service for fraud prediction."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import predict_transaction

app = FastAPI(
    title="FinTech Fraud Detection API",
    version="1.0.0",
    description="Predicts the probability that a synthetic card transaction is fraudulent.",
)


class Transaction(BaseModel):
    amount: float = Field(gt=0)
    hour: int = Field(ge=0, le=23)
    merchant_category: str
    country_mismatch: int = Field(ge=0, le=1)
    device_trust: float = Field(ge=0, le=1)
    transactions_24h: int = Field(ge=0)
    card_age_days: int = Field(gt=0)
    account_age_days: int = Field(gt=0)
    previous_chargebacks: int = Field(ge=0)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(transaction: Transaction):
    try:
        return predict_transaction(transaction.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
