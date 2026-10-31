"""
FastAPI scoring service for real-time anomaly detection.

Accepts sensor readings and returns anomaly scores and failure risk assessments.
"""

import os
import joblib
import numpy as np
from pathlib import Path
from typing import Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.features import FeatureBuffer, ANALOGUE_SIGNALS

# Model paths
MODEL_DIR = Path(__file__).parent.parent.parent / "models"
ISOLATION_FOREST_PATH = MODEL_DIR / "isolation_forest.joblib"
SCALER_PATH = MODEL_DIR / "iforest_scaler.joblib"
FEATURE_COLS_PATH = MODEL_DIR / "feature_cols.txt"

# Global model references
model = None
scaler = None
feature_cols = None
feature_buffer = None

# Simple detector parameters (based on training)
OIL_TEMP_BASELINE_MEAN = 59.7
OIL_TEMP_BASELINE_STD = 2.0
OIL_TEMP_THRESHOLD = OIL_TEMP_BASELINE_MEAN + 2 * OIL_TEMP_BASELINE_STD


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    global model, scaler, feature_cols, feature_buffer

    print("Loading models...")

    # Load Isolation Forest model
    if ISOLATION_FOREST_PATH.exists():
        model = joblib.load(ISOLATION_FOREST_PATH)
        print(f"Loaded Isolation Forest model from {ISOLATION_FOREST_PATH}")
    else:
        print(f"Warning: Model not found at {ISOLATION_FOREST_PATH}")

    # Load scaler
    if SCALER_PATH.exists():
        scaler = joblib.load(SCALER_PATH)
        print(f"Loaded scaler from {SCALER_PATH}")
    else:
        print(f"Warning: Scaler not found at {SCALER_PATH}")

    # Load feature columns
    if FEATURE_COLS_PATH.exists():
        with open(FEATURE_COLS_PATH) as f:
            feature_cols = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(feature_cols)} feature columns")
    else:
        print(f"Warning: Feature columns not found at {FEATURE_COLS_PATH}")

    # Initialize feature buffer
    feature_buffer = FeatureBuffer()
    print("Feature buffer initialized")

    print("Scoring service ready!")
    yield

    print("Shutting down...")


app = FastAPI(
    title="Metro Compressor Anomaly Detection API",
    description="Real-time anomaly detection for metro train air compressors",
    version="1.0.0",
    lifespan=lifespan
)


class SensorReading(BaseModel):
    """Input schema for sensor readings."""
    timestamp: Optional[str] = None
    TP2: float
    TP3: float
    H1: float
    DV_pressure: float
    Oil_temperature: float
    Motor_current: float
    # Digital signals (optional)
    COMP: Optional[int] = None
    DV_electric: Optional[int] = None
    Towers: Optional[int] = None
    MPG: Optional[int] = None
    LPS: Optional[int] = None
    Pressure_switch: Optional[int] = None
    Oil_level: Optional[int] = None
    Caudal_impulses: Optional[int] = None


class PredictionResponse(BaseModel):
    """Output schema for predictions."""
    timestamp: str
    is_anomaly: bool
    anomaly_score: float
    failure_risk: str
    simple_detector_alert: bool
    oil_temperature: float
    buffer_ready: bool


@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "service": "Metro Compressor Anomaly Detection",
        "status": "running",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }


@app.get("/health")
def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "feature_cols_loaded": feature_cols is not None,
        "buffer_ready": feature_buffer.is_ready() if feature_buffer else False
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading):
    """
    Process a sensor reading and return anomaly prediction.

    The service maintains a rolling buffer of readings to compute
    window features. Returns predictions once buffer is full.
    """
    global feature_buffer

    # Parse timestamp
    timestamp = reading.timestamp or datetime.now().isoformat()

    # Update feature buffer
    reading_dict = {
        'TP2': reading.TP2,
        'TP3': reading.TP3,
        'H1': reading.H1,
        'DV_pressure': reading.DV_pressure,
        'Oil_temperature': reading.Oil_temperature,
        'Motor_current': reading.Motor_current
    }
    feature_buffer.update(reading_dict)

    # Simple detector (always available)
    simple_alert = reading.Oil_temperature > OIL_TEMP_THRESHOLD

    # Default response if buffer not ready
    if not feature_buffer.is_ready():
        return PredictionResponse(
            timestamp=timestamp,
            is_anomaly=simple_alert,
            anomaly_score=0.0,
            failure_risk="unknown (buffer filling)",
            simple_detector_alert=simple_alert,
            oil_temperature=reading.Oil_temperature,
            buffer_ready=False
        )

    # Compute features
    if model is None or scaler is None or feature_cols is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Check model files in /models directory."
        )

    feature_vector = feature_buffer.get_feature_vector(feature_cols)
    feature_scaled = scaler.transform(feature_vector.reshape(1, -1))

    # Get anomaly score from Isolation Forest
    # score_samples returns negative values, more negative = more anomalous
    raw_score = model.score_samples(feature_scaled)[0]
    # Convert to positive anomaly score (higher = more anomalous)
    anomaly_score = -raw_score

    # Predict anomaly
    prediction = model.predict(feature_scaled)[0]
    is_anomaly = prediction == -1  # Isolation Forest: -1 = anomaly

    # Combine with simple detector
    is_anomaly = is_anomaly or simple_alert

    # Determine risk level
    if anomaly_score > 0.6 or simple_alert:
        risk = "HIGH"
    elif anomaly_score > 0.5:
        risk = "MEDIUM"
    elif anomaly_score > 0.4:
        risk = "LOW"
    else:
        risk = "NORMAL"

    return PredictionResponse(
        timestamp=timestamp,
        is_anomaly=is_anomaly,
        anomaly_score=float(anomaly_score),
        failure_risk=risk,
        simple_detector_alert=simple_alert,
        oil_temperature=reading.Oil_temperature,
        buffer_ready=True
    )


@app.post("/reset")
def reset_buffer():
    """Reset the feature buffer (use when starting a new stream)."""
    global feature_buffer
    feature_buffer = FeatureBuffer()
    return {"status": "buffer reset"}


@app.get("/model/info")
def model_info():
    """Get information about the loaded model."""
    if model is None:
        return {"error": "Model not loaded"}

    return {
        "model_type": type(model).__name__,
        "n_features": len(feature_cols) if feature_cols else 0,
        "feature_columns": feature_cols,
        "oil_temp_threshold": OIL_TEMP_THRESHOLD,
        "contamination": getattr(model, 'contamination', 'N/A')
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
