"""
Urban Traffic Prediction & Congestion Analysis System
Module: Prediction & Inference (prediction.py)

This module handles:
1. Loading the serialized scikit-learn model pipeline from models/best_model.pkl.
2. Validating input parameters (range bounds, non-negative values).
3. Applying feature engineering to match the training pipeline schema.
4. Returning predicted congestion level, traffic severity category, and Traffic Stress Index (TSI).
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

# Add src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from feature_engineering import (
    add_derived_features,
    classify_congestion,
    calculate_rain_congestion_adjustment,
    compute_traffic_stress_index,
    classify_tsi,
    get_feature_lists
)


def get_model_path(filename="best_model.pkl"):
    """
    Locates the serialized model pipeline whether called from root or src.
    """
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_file_dir) if os.path.basename(current_file_dir) == "src" else current_file_dir

    possible_paths = [
        os.path.join(project_root, "models", filename),
        os.path.join("models", filename),
        os.path.join("..", "models", filename),
        filename,
        os.path.join("..", filename)
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    raise FileNotFoundError(f"Trained model not found at: {filename}. Please run train_models.py first.")



# Cache loaded model in memory for fast inference
_LOADED_MODEL = None

def load_prediction_model(model_path=None):
    """
    Loads and caches the serialized model pipeline.
    """
    global _LOADED_MODEL
    if _LOADED_MODEL is None:
        if model_path is None:
            model_path = get_model_path("best_model.pkl")
        print(f"[INFO] Loading trained model from: {model_path}")
        _LOADED_MODEL = joblib.load(model_path)
    return _LOADED_MODEL


def validate_input_dict(input_data):
    """
    Validates input ranges and sets standard defaults for non-provided features.
    """
    # Required core keys
    numeric_bounds = {
        "traffic_volume": (0, 15000),
        "average_speed": (1.0, 120.0),
        "travel_time_index": (0.5, 10.0),
        "road_capacity_utilization": (1.0, 150.0),
        "incident_reports": (0, 50),
        "environmental_impact": (0.0, 100.0),
        "public_transport_usage": (0.0, 100.0),
        "traffic_signal_compliance": (0.0, 100.0),
        "parking_usage": (0.0, 100.0),
        "pedestrian_cyclist_count": (0, 5000),
        "temp_min": (5.0, 45.0),
        "temp_max": (10.0, 50.0),
        "temp_mean": (10.0, 45.0),
        "rain_mm": (0.0, 300.0),
        "wind_speed": (0.0, 80.0),
        "weekday": (0, 6),
        "is_peak": (0, 1)
    }

    clean_dict = {}
    for key, (min_val, max_val) in numeric_bounds.items():
        val = input_data.get(key, None)
        if val is None:
            # Provide sensible median defaults if missing
            default_map = {
                "traffic_volume": 2500,
                "average_speed": 25.0,
                "travel_time_index": 2.2,
                "road_capacity_utilization": 80.0,
                "incident_reports": 1,
                "environmental_impact": 50.0,
                "public_transport_usage": 45.0,
                "traffic_signal_compliance": 85.0,
                "parking_usage": 70.0,
                "pedestrian_cyclist_count": 150,
                "temp_min": 20.0,
                "temp_max": 30.0,
                "temp_mean": 25.0,
                "rain_mm": 0.0,
                "wind_speed": 12.0,
                "weekday": 2,
                "is_peak": 1
            }
            val = default_map.get(key, 0.0)
        else:
            val = float(val)
            if val < 0 and min_val >= 0:
                raise ValueError(f"Feature '{key}' cannot be negative (got {val}).")
            val = float(np.clip(val, min_val, max_val))
        clean_dict[key] = val

    # Categorical validation
    clean_dict["area_name"] = str(input_data.get("area_name", "Indiranagar")).strip()
    clean_dict["road_name"] = str(input_data.get("road_name", "100 Feet Road")).strip()
    clean_dict["weather_conditions"] = str(input_data.get("weather_conditions", "Clear")).strip()
    clean_dict["roadwork_activity"] = str(input_data.get("roadwork_activity", "No")).strip()

    return clean_dict


def predict_congestion(input_data, model=None):
    """
    Executes end-to-end prediction for a single input dictionary:
    1. Validates and cleans input values.
    2. Constructs DataFrame and applies feature engineering.
    3. Runs model inference.
    4. Computes post-prediction Congestion Category & Traffic Stress Index (TSI).
    """
    if model is None:
        model = load_prediction_model()

    # 1. Validate inputs
    clean_dict = validate_input_dict(input_data)

    # 2. Build single-row DataFrame
    df_single = pd.DataFrame([clean_dict])

    # 3. Add derived features (traffic_intensity, incident_density, is_weekend)
    df_single = add_derived_features(df_single)

    # 4. Extract ML feature columns in exact order
    cat_cols, num_cols, _, _ = get_feature_lists()
    feature_cols = cat_cols + num_cols
    X_input = df_single[feature_cols].copy()

    # 5. Run inference
    raw_pred = model.predict(X_input)[0]
    peak_adjustment = 1.15 if int(clean_dict.get("is_peak", 0)) else 1.0
    rain_adjustment = calculate_rain_congestion_adjustment(clean_dict.get("rain_mm", 0.0))
    predicted_congestion = float(np.clip(
        (raw_pred * peak_adjustment) + rain_adjustment,
        0.0,
        100.0
    ))

    # 6. Post-prediction classifications
    cong_category, cong_color, cong_desc = classify_congestion(predicted_congestion)
    
    # 7. Compute Traffic Stress Index (TSI)
    tsi_score = compute_traffic_stress_index(
        congestion_level=predicted_congestion,
        is_peak=int(clean_dict.get("is_peak", 0)),
        rain_mm=float(clean_dict.get("rain_mm", 0.0)),
        wind_speed=float(clean_dict.get("wind_speed", 0.0))
    )
    tsi_category, tsi_color = classify_tsi(tsi_score)

    return {
        "predicted_congestion_level": round(predicted_congestion, 2),
        "congestion_category": cong_category,
        "congestion_color": cong_color,
        "congestion_description": cong_desc,
        "traffic_stress_index": tsi_score,
        "tsi_category": tsi_category,
        "tsi_color": tsi_color,
        "input_summary": clean_dict
    }


if __name__ == "__main__":
    # Quick sanity test with realistic Bengaluru corridor inputs
    sample_input = {
        "area_name": "Hebbal",
        "road_name": "Hebbal Flyover",
        "traffic_volume": 4200,
        "average_speed": 14.5,
        "travel_time_index": 3.8,
        "road_capacity_utilization": 96.0,
        "incident_reports": 2,
        "environmental_impact": 68.0,
        "public_transport_usage": 55.0,
        "traffic_signal_compliance": 82.0,
        "parking_usage": 88.0,
        "pedestrian_cyclist_count": 120,
        "weather_conditions": "Rain",
        "roadwork_activity": "Yes",
        "temp_min": 21.0,
        "temp_max": 28.0,
        "temp_mean": 24.5,
        "rain_mm": 18.5,
        "wind_speed": 16.0,
        "weekday": 4, # Friday
        "is_peak": 1   # Peak hour
    }
    
    print("\n--- Testing Prediction Module ---")
    result = predict_congestion(sample_input)
    print("Prediction Result:")
    for k, v in result.items():
        if k != "input_summary":
            print(f"  {k}: {v}")
