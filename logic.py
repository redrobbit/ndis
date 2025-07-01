import numpy as np
import pandas as pd
import math

sensor_default_area = {
    "Seismic": (None, None),
    "Magnetometers": (500, 200),
    "Lidar": (400, 400),
    "GPR": (None, None),
    "Camera": (300, 300),
    "Thermal_Camera": (300, 300),
    "Hyperspectral": (1000, 200),
    "Multispectral": (1000, 200),
    "EM": (400, 400),
    "Gravimeter": (500, 500)
}

sensor_spacing = {
    "Magnetometers": 5,
    "Lidar": 10,
    "EM": 10,
    "Gravimeter": 10,
    "Hyperspectral": 20,
    "Multispectral": 20,
    "Camera": 20,
    "Thermal_Camera": 20
}

hazard_stage_survey_map = {
    ("Volcano", "pre_event"): ["Magnetometers", "Seismic", "Camera"],
    ("Volcano", "during"): ["Thermal_Camera", "Camera", "Lidar"],
    ("Volcano", "post_event"): ["Lidar", "Camera", "Seismic"],
    ("Volcano", "clean_up"): ["Lidar", "Camera", "Seismic"],
    ("Earthquake", "pre_event"): ["Seismic", "Magnetometers", "Camera"],
    ("Earthquake", "during"): ["Seismic", "Camera", "Lidar"],
    ("Earthquake", "post_event"): ["Lidar", "Camera", "Seismic"],
    ("Fault", "pre_event"): ["Seismic", "Magnetometers", "Camera"],
    ("Fault", "post_event"): ["Seismic", "Camera", "Lidar"],
    ("Landslide", "pre_event"): ["Lidar", "GPR", "Camera"],
    ("Landslide", "during"): ["Camera", "Thermal_Camera", "Lidar"],
    ("Landslide", "post_event"): ["Lidar", "Seismic", "Camera"],
    ("Landslide", "clean_up"): ["Camera", "Lidar", "Seismic"],
    ("Tsunami", "during"): ["Camera", "Thermal_Camera", "Lidar"],
    ("Tsunami", "post_event"): ["Camera", "Lidar", "Seismic"],
    ("Tsunami", "clean_up"): ["Camera", "Lidar", "Seismic"],
    ("Nuclear", "pre_event"): ["Thermal_Camera", "Camera", "Lidar"],
    ("Nuclear", "during"): ["Thermal_Camera", "Camera", "Lidar"],
    ("Nuclear", "post_event"): ["Camera", "Lidar", "Seismic"],
    ("Nuclear", "clean_up"): ["Camera", "Lidar", "Seismic"]
}

comm_range_bins = np.array([300, 500, 1000, 2000, 3000, 5000, 7000, 10000, 15000, 20000, 45000, 100000, 200000])
distance_range_bins = np.array([21, 300, 700, 1000, 2000, 3000, 5000, 7000, 10000, 15000, 20000, 30000, 50000, 100000, 1046000])

def get_bin(value, bins):
    return np.digitize(value, bins)

def compute_mission_flight_distance(sensor_name, hazard_distance, area_length=None, area_width=None):
    if sensor_name == "Seismic":
        return hazard_distance
    elif sensor_name == "GPR":
        return 1000
    if area_length and area_width:
        spacing = sensor_spacing.get(sensor_name, 10)
        num_lines = math.ceil(area_width / spacing)
        return num_lines * area_length
    else:
        area_length, area_width = sensor_default_area.get(sensor_name, (None, None))
        if area_length and area_width:
            spacing = sensor_spacing.get(sensor_name, 10)
            num_lines = math.ceil(area_width / spacing)
            return num_lines * area_length
        else:
            if sensor_name == "Magnetometers":
                return 20000
            elif sensor_name == "Lidar":
                return math.pi * hazard_distance
            else:
                return hazard_distance

def full_mission_recommender_api(input_dict, drone_df, sensor_df, ghz_df):
    try:
        print("🚀 Mission Input:", input_dict)
        hazard_type = input_dict.get("geohazard_type")
        stage = input_dict.get("hazard_stage")
        sensor_override = input_dict.get("sensor")
        area_length = input_dict.get("area_length")
        area_width = input_dict.get("area_width")
        lat = float(input_dict.get("latitude"))
        lon = float(input_dict.get("longitude"))

        # Force numeric conversion of GHZ coordinates
        ghz_df["latitude"] = pd.to_numeric(ghz_df["latitude"], errors="coerce")
        ghz_df["longitude"] = pd.to_numeric(ghz_df["longitude"], errors="coerce")
        ghz_df = ghz_df.dropna(subset=["latitude", "longitude"])

        ghz_df["dist_to_input"] = (ghz_df["latitude"] - lat)**2 + (ghz_df["longitude"] - lon)**2
        ghz_df = ghz_df.sort_values(by="dist_to_input")
        target_row = ghz_df.iloc[0]
        hazard_distance = float(target_row["distance"])

        sensors_to_use = sensor_override or hazard_stage_survey_map.get((hazard_type, stage), ["Seismic", "Camera", "Lidar"])
        if isinstance(sensors_to_use, str):
            sensors_to_use = [sensors_to_use]

        sensor_weight_map = sensor_df.set_index("sensor_name")["sensor_weight"].to_dict()
        output_results = []

        for sensor_name in sensors_to_use:
            sensor_weight = sensor_weight_map.get(sensor_name)
            flight_path_distance = compute_mission_flight_distance(sensor_name, hazard_distance, area_length, area_width)
            eligible = drone_df if sensor_name == "Camera" else drone_df[drone_df["max_payload_weight"] >= sensor_weight].copy()

            if sensor_name == "Seismic":
                hazard_bin = get_bin(hazard_distance, comm_range_bins)
                eligible = eligible[eligible["comm_range"].apply(lambda cr: get_bin(cr, comm_range_bins) >= hazard_bin)].copy()
            else:
                hazard_bin = get_bin(flight_path_distance, distance_range_bins)
                eligible = eligible[eligible["distance_range"].apply(lambda dr: get_bin(dr, distance_range_bins) >= hazard_bin)].copy()

            if eligible.empty:
                eligible = drone_df if sensor_name == "Camera" else drone_df[drone_df["max_payload_weight"] >= sensor_weight].copy()

            eligible["payload_overkill"] = (eligible["max_payload_weight"] - sensor_weight).abs()
            eligible["distance_overkill"] = (
                (eligible["comm_range"] - hazard_distance).abs() if sensor_name == "Seismic" 
                else (eligible["distance_range"] - flight_path_distance).abs()
            )
            eligible["score"] = (eligible["payload_overkill"] * 0.5) + (eligible["distance_overkill"] * 0.5)
            sorted_drones = eligible.sort_values(by="score")
            top3_drones = sorted_drones.head(3)

            output_results.append({
                "sensor": sensor_name,
                "flight_path_m": flight_path_distance,
                "top3_drones": top3_drones[["mfc_model", "max_payload_weight", "distance_range", "flight_time", "comm_range"]].to_dict(orient="records")
            })

        flattened = []
        for res in output_results:
            for d in res["top3_drones"]:
                flattened.append({**d, "sensor": res["sensor"], "flight_path_m": res["flight_path_m"]})
        best_combo = min(flattened, key=lambda x: (x["max_payload_weight"], x["distance_range"]))

        logistics = "Mission may require offshore support." if hazard_distance > 50000 else "Standard deployment possible."

        return {
            "mission_summary": output_results,
            "best_combo": best_combo,
            "logistics": logistics
        }

    except Exception as e:
        print("❌ ERROR:", str(e))
        return {"error": f"Server error: {str(e)}"}
