import math
import random
import pandas as pd
import numpy as np

# ----------------------------
# Canonical sensor names & helpers
# ----------------------------
_CANON = {
    "lidar": "LiDAR",
    "thermal camera": "Thermal Camera",
    "gamma spectrometer": "Gamma Spectrometer",
    "magnetometer": "Magnetometers",
    "magnetometers": "Magnetometers",
    "gpr": "GPR",
    "bpr": "BPR",
    "camera": "Camera",
    "seismic": "Seismic",
    "multispectral": "Multispectral",
    "hyperspectral": "Hyperspectral",
    "em": "EM",
    "gravimeter": "Gravimeter",
}
def canon_sensor(name: str) -> str:
    if not name:
        return ""
    key = str(name).strip().lower()
    return _CANON.get(key, name)

# ----------------------------
# Disaster phase → sensor map (revised)
# ----------------------------
DISASTER_MAP = [
    ("Volcano", "Pre-Event", ["Magnetometers", "Seismic", "Camera"]),
    ("Volcano", "During", ["Thermal Camera", "Camera", "LiDAR"]),
    ("Volcano", "Post-Event", ["LiDAR", "Camera", "Seismic"]),
    ("Volcano", "Clean-Up", ["LiDAR", "Camera", "Seismic"]),
    ("Earthquake", "Pre-Event", ["Seismic", "Magnetometers", "Camera"]),
    ("Earthquake", "During", ["Seismic", "Camera", "LiDAR"]),
    ("Earthquake", "Post-Event", ["LiDAR", "Camera", "Seismic"]),
    ("Fault", "Pre-Event", ["Seismic", "Magnetometers", "Camera"]),
    ("Fault", "Post-Event", ["Seismic", "Camera", "LiDAR"]),
    ("Landslide", "Pre-Event", ["LiDAR", "GPR", "Camera"]),
    ("Landslide", "During", ["Camera", "Thermal Camera", "LiDAR"]),
    ("Landslide", "Post-Event", ["LiDAR", "Seismic", "Camera"]),
    ("Landslide", "Clean-Up", ["Camera", "LiDAR", "Seismic"]),
    ("Tsunami", "During", ["BPR", "Camera", "Seismic"]),
    ("Tsunami", "Post-Event", ["BPR", "Camera", "LiDAR"]),
    ("Tsunami", "Clean-Up", ["Camera", "LiDAR", "Thermal Camera"]),
    ("Nuclear", "Pre-Event", ["Thermal Camera", "Camera", "LiDAR"]),
    ("Nuclear", "During", ["Thermal Camera", "Camera", "LiDAR"]),
    ("Nuclear", "Post-Event", ["Camera", "LiDAR", "Gamma Spectrometer"]),
    ("Nuclear", "Clean-Up", ["Camera", "LiDAR", "Gamma Spectrometer"]),
]
DISASTER_LOOKUP = {(h, s): v for (h, s, v) in DISASTER_MAP}

# ----------------------------
# Defaults for mission distance (when user didn't specify mapping area)
# ----------------------------
SENSOR_DEFAULT_AREA = {
    "Seismic": (None, None),
    "Magnetometers": (500, 200),
    "LiDAR": (400, 400),
    "GPR": (None, None),
    "Camera": (300, 300),
    "Thermal Camera": (300, 300),
    "Hyperspectral": (1000, 200),
    "Multispectral": (1000, 200),
    "EM": (400, 400),
    "Gravimeter": (500, 500),
    "Gamma Spectrometer": (500, 500),
    "BPR": (None, None),
}
SENSOR_SPACING = {
    "Magnetometers": 5,
    "LiDAR": 10,
    "EM": 10,
    "Gravimeter": 10,
    "Hyperspectral": 20,
    "Multispectral": 20,
    "Camera": 20,
    "Thermal Camera": 20,
    "Gamma Spectrometer": 20,
}
DIRECT_DELIVERY = {"Seismic", "GPR", "BPR"}

# ----------------------------
# Mission distance calculator
# ----------------------------
def compute_mission_distance(sensor: str, distance_to_road: float,
                             area_length: float | None,
                             area_width: float | None,
                             spacing: float | None) -> tuple[float, str, str]:
    """
    Returns (mission_distance_m, mission_type, mission_text)
    """
    s = canon_sensor(sensor)

    # DELIVERY missions
    if s in DIRECT_DELIVERY:
        md = float(distance_to_road or 0)
        text = (
            f"{s} mission is classified as Delivery. "
            f"Max effective distance from GCS equals the platform communication range; "
            f"travel distance fallback uses distance_to_road input ({int(md)} m)."
        )
        return md, "Delivery", text

    # MAPPING missions (explicit user area)
    if area_length and area_width and spacing and spacing > 0:
        num_lines = math.ceil(float(area_width) / float(spacing))
        md = float(num_lines) * float(area_length)
        text = (
            f"{s} mission is classified as Mapping. Flight path derived from area and line spacing "
            f"({int(area_length)}×{int(area_width)} m @ {int(spacing)} m) totals ≈{int(md)} m."
        )
        return md, "Mapping", text

    # Fallbacks by sensor defaults
    length, width = SENSOR_DEFAULT_AREA.get(s, (None, None))
    if length and width:
        spacing_eff = SENSOR_SPACING.get(s, 10)
        num_lines = math.ceil(width / spacing_eff)
        md = num_lines * length
        text = (
            f"{s} mapping uses default area {length}×{width} m with {spacing_eff} m spacing → "
            f"≈{int(md)} m flight path."
        )
        return md, "Mapping", text

    if s == "Magnetometers":
        md = 20000.0
        return md, "Mapping", "Magnetometers default mapping path ≈20,000 m (line survey heuristic)."
    if s == "LiDAR":
        radius = max(float(distance_to_road or 1000), 1000.0)
        md = math.pi * radius
        return md, "Mapping", f"LiDAR perimeter heuristic π·R with R≈{int(radius)} m → ≈{int(md)} m."
    # Default fallback
    md = float(distance_to_road or 1000.0)
    return md, "Mapping", f"Fallback mapping path uses access distance ≈{int(md)} m."

# ----------------------------
# Drone ranking (single-request scorer)
# ----------------------------
def score_and_pick_drones(drone_df: pd.DataFrame,
                          sensor_weight_g: float,
                          mission_distance_m: float,
                          access_distance_m: float,
                          top_n: int = 3) -> list[dict]:
    df = drone_df.copy()

    # numeric coerce
    for col in ["comm_range", "distance_range", "max_payload_weight", "flight_time", "price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # payload filter
    sw = float(sensor_weight_g or 0)
    df["max_payload_weight"] = df["max_payload_weight"].fillna(0)
    df = df[df["max_payload_weight"] >= sw]
    if df.empty:
        return []

    def _row_score(r):
        score = 100.0
        note = ""
        # comm check vs access
        if not pd.isna(r.get("comm_range")) and r["comm_range"] < access_distance_m:
            score += 30; note = "Comm range insufficient"
        else:
            score -= 5; note = "Comm OK"

        # mission coverage
        dr = r.get("distance_range", np.nan)
        if pd.isna(dr):
            score += 10; note += " · Distance unknown"
            swaths = None
        else:
            swaths = 1 if mission_distance_m <= 0 else max(1, math.ceil(mission_distance_m / max(dr, 1)))
            if dr >= mission_distance_m:
                score -= 10; note += " · Full coverage"
            elif dr >= 0.75 * mission_distance_m:
                score += 2; note += " · Near full coverage"
            elif dr >= 0.5 * mission_distance_m:
                score += 5; note += " · 2–3 swaths needed"
            elif dr >= 0.25 * mission_distance_m:
                score += 15; note += " · Multiple passes"
            else:
                score += 25; note += " · Very limited range"

        # payload headroom mild penalty
        payload_diff = (r["max_payload_weight"] - sw)
        score += min(abs(payload_diff) * 0.01, 10)

        # jitter to break ties (small)
        score += random.uniform(-0.15, 0.15)
        return pd.Series([score, note, swaths])

    df[["score", "note", "swaths_needed"]] = df.apply(_row_score, axis=1)

    # select fields for UI (ensure columns)
    cols = [
        "mfc_model","manufacturer","configuration_harmonized",
        "comm_range","distance_range","max_payload_weight","flight_time",
        "price","image","source"
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None

    out = df.sort_values("score").drop_duplicates("mfc_model").head(top_n).copy()
    for c in ["comm_range","distance_range","max_payload_weight","flight_time","price"]:
        if c in out:
            out[c] = out[c].apply(lambda x: None if pd.isna(x) else float(x))
    out["note"] = out["note"].astype(str)

    return out[cols + ["swaths_needed", "note"]].to_dict(orient="records")

# ----------------------------
# API entry
# ----------------------------
def mission_recommender(input_dict: dict,
                        drone_df: pd.DataFrame,
                        sensor_df: pd.DataFrame) -> dict:
    """
    input_dict keys:
      distance_to_road, area_length, area_width, spacing,
      geohazard_type, hazard_stage, sensor (optional)
    """
    hazard = input_dict.get("geohazard_type")
    stage = input_dict.get("hazard_stage")
    sensor_override = input_dict.get("sensor") or ""
    distance_to_road = float(input_dict.get("distance_to_road") or 0)
    area_length = input_dict.get("area_length")
    area_width = input_dict.get("area_width")
    spacing = input_dict.get("spacing")

    # numeric coercion for overrides
    area_length = float(area_length) if area_length not in (None, "") else None
    area_width = float(area_width) if area_width not in (None, "") else None
    spacing = float(spacing) if spacing not in (None, "") else None

    # Determine sensors to use
    if sensor_override:
        sensors_to_use = [canon_sensor(sensor_override)]
        top3_sensors = sensors_to_use
    else:
        sensors_to_use = DISASTER_LOOKUP.get((hazard, stage), ["Camera"])
        sensors_to_use = [canon_sensor(s) for s in sensors_to_use]
        top3_sensors = sensors_to_use[:3]

    # Build sensor weight + meta lookup (matches your CSV columns)
    s_meta = sensor_df.copy()
    s_meta.columns = [str(c).strip() for c in s_meta.columns]
    if "sensor_name" not in s_meta.columns:
        raise ValueError("sensor_df is missing 'sensor_name' column")
    s_meta["sensor_name"] = s_meta["sensor_name"].map(canon_sensor)
    s_meta = s_meta.set_index("sensor_name")
    sensor_weight_map = s_meta["sensor_weight"].to_dict() if "sensor_weight" in s_meta.columns else {}

    mission_sections = []
    best_candidates_flat = []

    for s in sensors_to_use:
        sw = float(sensor_weight_map.get(s, 0))
        md, mtype, mtext = compute_mission_distance(
            s, distance_to_road, area_length, area_width, spacing
        )

        # Drone candidates
        cands = score_and_pick_drones(
            drone_df=drone_df,
            sensor_weight_g=sw,
            mission_distance_m=md,
            access_distance_m=distance_to_road,
            top_n=3
        )

        # annotate max gcs distance for each drone (its comm_range)
        for d in cands:
            d["max_gcs_distance_m"] = d.get("comm_range")

        # sensor meta slice for UI
        sm = {}
        if s in s_meta.index:
            sm = {
                "model": (s_meta.at[s, "model"] if "model" in s_meta.columns else None),
                "sensor_weight": (float(s_meta.at[s, "sensor_weight"]) if "sensor_weight" in s_meta.columns else None),
                "source": (s_meta.at[s, "source"] if "source" in s_meta.columns else None),
            }

        mission_sections.append({
            "sensor": s,
            "mission_type": mtype,
            "mission_text": mtext,
            "mission_distance_m": float(md),
            "sensor_meta": sm,
            "top3_drones": cands
        })

        for d in cands:
            best_candidates_flat.append({**d, "sensor": s, "mission_distance_m": md})

    # pick a "best combo"
    best_combo = None
    if mission_sections and mission_sections[0]["top3_drones"]:
        best = mission_sections[0]["top3_drones"][0]
        best_combo = {**best, "sensor": mission_sections[0]["sensor"],
                      "swaths_needed": best.get("swaths_needed"),
                      "max_gcs_distance_m": best.get("max_gcs_distance_m")}
    elif best_candidates_flat:
        best_combo = best_candidates_flat[0]

    # Mission overview string
    s_list_text = sensor_override if sensor_override else ", ".join(top3_sensors)
    mission_overview = (
        f"Hazard: {hazard} · Stage: {stage}. "
        f"Candidate sensors: {s_list_text}. "
        f"Access distance (to base/road): {int(distance_to_road)} m."
    )

    logistics = "Standard deployment possible."
    if distance_to_road > 50000:
        logistics = "Long-range access—consider forward base or offshore/remote staging."

    return {
        "mission_overview": mission_overview,
        "top3_sensors": top3_sensors,
        "mission_summary": mission_sections,
        "best_combo": best_combo or {},
        "logistics": logistics
    }
