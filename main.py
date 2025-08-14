from pathlib import Path
import math
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from logic import mission_recommender

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

def load_csv_numeric(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# Load CSVs using absolute paths
drone_df = pd.read_csv(BASE_DIR / "drone_df.csv", dtype=str)
drone_df = load_csv_numeric(drone_df, ["comm_range","distance_range","max_payload_weight","flight_time","price"])

sensor_df = pd.read_csv(BASE_DIR / "sensor_df.csv", dtype=str)
sensor_df = load_csv_numeric(sensor_df, ["sensor_weight"])

class MissionInput(BaseModel):
    distance_to_road: float
    area_length: float | None = None
    area_width: float | None = None
    spacing: float | None = None
    geohazard_type: str
    hazard_stage: str
    sensor: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    radius: float | None = None

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# -------------- JSON sanitizers (avoid NaN/Inf in responses) --------------
def _finite_float(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return v

def _sanitize_json(obj):
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        return x if math.isfinite(x) else None
    return obj

# -------------- Global error handler to always return JSON --------------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # prints show up in App Service log stream
    print(f"UNHANDLED ERROR: {exc}")
    return JSONResponse(status_code=500, content={"error": f"Server error: {str(exc)}"})

@app.post("/mission")
def mission_api(input_data: MissionInput):
    payload = input_data.dict()
    result = mission_recommender(payload, drone_df, sensor_df)

    # Best combo known numeric fields → finite
    if isinstance(result.get("best_combo"), dict):
        for k in ["comm_range","distance_range","max_payload_weight","flight_time","price","swaths_needed","max_gcs_distance_m"]:
            if k in result["best_combo"]:
                result["best_combo"][k] = _finite_float(result["best_combo"][k])

    # Sanitize entire result
    result = _sanitize_json(result)
    return result
