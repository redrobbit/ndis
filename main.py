from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
from logic import full_mission_recommender_api

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Clean CSV loader to force numeric columns
def load_clean_csv(file, numeric_cols):
    df = pd.read_csv(file, dtype=str)
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

# Load datasets
drone_df = load_clean_csv("drone_df.csv", numeric_cols=[
    "max_payload_weight", "distance_range", "flight_time", "comm_range"
])
sensor_df = pd.read_csv("sensor_df.csv")
ghz_df = pd.read_csv("ghz_df.csv")

stage_mapping = {
    "Pre-event": "pre_event",
    "During": "during",
    "Post-event": "post_event",
    "Clean-up": "clean_up"
}

class MissionInput(BaseModel):
    latitude: float = None
    longitude: float = None
    radius: float
    area_length: float = None
    area_width: float = None
    geohazard_type: str
    hazard_stage: str
    sensor: str = None

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/mission")
def mission_api(input_data: MissionInput):
    try:
        if input_data.latitude is None or input_data.longitude is None:
            return {"error": "Latitude and longitude are required."}

        stage = stage_mapping.get(input_data.hazard_stage, input_data.hazard_stage)

        input_dict = {
            "latitude": input_data.latitude,
            "longitude": input_data.longitude,
            "radius": input_data.radius,
            "geohazard_type": input_data.geohazard_type,
            "hazard_stage": stage,
            "sensor": input_data.sensor,
            "area_length": input_data.area_length,
            "area_width": input_data.area_width
        }

        print("🚀 Mission Input:", input_dict)

        result = full_mission_recommender_api(input_dict, drone_df, sensor_df, ghz_df)
        return result

    except Exception as e:
        print("❌ ERROR:", e)
        return {"error": f"Server error: {str(e)}"}
