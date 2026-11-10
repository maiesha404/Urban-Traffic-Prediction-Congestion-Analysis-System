"""
Urban Traffic Prediction & Congestion Analysis System
Module: Feature Engineering (feature_engineering.py)

This module handles:
1. Deriving temporal features (month, is_weekend).
2. Deriving domain-specific traffic dynamics (traffic_intensity, incident_density).
3. Target Leakage Prevention: Explicitly separating model input features from
   post-prediction display metrics (Congestion Level, Traffic Stress Index).
4. Categorization helpers for Congestion and Traffic Stress Index (TSI).
"""

import pandas as pd
import numpy as np


def add_derived_features(df):
    """
    Computes additional derived features from time and traffic metrics:
    - month: Calendar month (1 to 12)
    - is_weekend: Boolean flag (1 if Saturday or Sunday, else 0)
    - traffic_intensity: Traffic Volume relative to Road Capacity Utilization
    - incident_density: Incident Reports per unit of Traffic Volume
    """
    df = df.copy()

    # 1. Temporal derivations
    if "time" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"])
        df["month"] = df["time"].dt.month
        # If weekday is not already present, calculate it (0=Mon, 6=Sun)
        if "weekday" not in df.columns:
            df["weekday"] = df["time"].dt.weekday
    elif "weekday" not in df.columns:
        df["weekday"] = 0

    # is_weekend: Saturday (5) and Sunday (6)
    if "weekday" in df.columns and "is_weekend" not in df.columns:
        df["is_weekend"] = df["weekday"].apply(lambda w: 1 if w in [5, 6] else 0)

    # 2. Traffic-derived ratios (safely avoiding division by zero)
    if "traffic_volume" in df.columns and "road_capacity_utilization" in df.columns:
        # Utilization represents capacity percentage (e.g., 50 to 100)
        df["traffic_intensity"] = df["traffic_volume"] / (df["road_capacity_utilization"] + 1e-5)
    else:
        df["traffic_intensity"] = 0.0

    if "incident_reports" in df.columns and "traffic_volume" in df.columns:
        df["incident_density"] = df["incident_reports"] / (df["traffic_volume"] + 1.0)
    else:
        df["incident_density"] = 0.0

    return df


def calculate_weather_severity(rain_mm, wind_speed):
    """
    Calculates composite weather severity factor based on precipitation and wind.
    Reference baseline: normalized impact >= 1.0.
    """
    # Normalized rainfall and wind severity scaling
    rain_factor = float(np.clip(rain_mm / 25.0, 0.0, 2.5))
    wind_factor = float(np.clip(wind_speed / 20.0, 0.0, 1.5))
    # Base multiplier is 1.0 (clear weather baseline)
    severity = 1.0 + (0.5 * rain_factor) + (0.3 * wind_factor)
    return float(severity)


def calculate_rain_congestion_adjustment(rain_mm):
    """Returns the congestion percentage-point increase caused by rainfall."""
    rain_value = max(float(rain_mm), 0.0)
    return float(np.clip(rain_value * 0.5, 0.0, 30.0))


def compute_traffic_stress_index(congestion_level, is_peak=0, rain_mm=0.0, wind_speed=0.0):
    """
    Calculates the post-prediction Traffic Stress Index (TSI).
    
    Formula:
        TSI = Congestion_Level * Peak_Weight * Weather_Severity
        
    Note:
        This is an operational analytical metric calculated AFTER prediction.
        It is NEVER used as an input feature during model training.
    """
    peak_weight = 1.35 if is_peak else 1.0
    weather_severity = calculate_weather_severity(rain_mm, wind_speed)
    tsi_value = float(congestion_level) * peak_weight * weather_severity
    return round(tsi_value, 2)


def classify_congestion(congestion_level):
    """
    Maps continuous congestion percentage (0-100) to severity tiers.
    """
    val = float(congestion_level)
    if val < 40.0:
        return "Low", "emerald", "Normal flow, negligible delays"
    elif val < 65.0:
        return "Moderate", "amber", "Moderate volume, minor delays at signals"
    elif val < 85.0:
        return "High", "orange", "Heavy congestion, significant slowdowns"
    elif val < 95.0:
        return "Very High", "rose", "Severe gridlock, bottlenecked intersections"
    else:
        return "Critical", "red", "Critical gridlock, standstill traffic"


def classify_tsi(tsi_value):
    """
    Maps Traffic Stress Index (TSI) value to operational threat levels based on quantiles.
    """
    val = float(tsi_value)
    if val < 90.0:
        return "LOW", "emerald"
    elif val < 140.0:
        return "MODERATE", "amber"
    elif val < 200.0:
        return "HIGH", "orange"
    elif val < 260.0:
        return "VERY HIGH", "rose"
    else:
        return "CRITICAL", "red"


def get_feature_lists():
    """
    Defines the exact feature columns used for Machine Learning.
    
    Target Leakage Prevention Rationale:
    1. 'congestion_level' is the regression TARGET variable.
    2. 'traffic_stress_index' is computed directly from congestion_level.
    3. 'peak_weight', 'rain_norm', 'wind_norm', 'weather_severity' are derived 
       weather/peak factors that are redundant with raw weather columns and is_peak.
    4. 'time' is the timestamp identifier (date) and not a direct tabular feature.
    """
    categorical_features = [
        "area_name",
        "road_name",
        "weather_conditions",
        "roadwork_activity"
    ]

    numerical_features = [
        "traffic_volume",
        "average_speed",
        "travel_time_index",
        "road_capacity_utilization",
        "incident_reports",
        "environmental_impact",
        "public_transport_usage",
        "traffic_signal_compliance",
        "parking_usage",
        "pedestrian_cyclist_count",
        "temp_min",
        "temp_max",
        "temp_mean",
        "rain_mm",
        "wind_speed",
        "weekday",
        "is_peak",
        "is_weekend",
        "traffic_intensity",
        "incident_density"
    ]

    target_col = "congestion_level"

    # Explicitly forbidden columns (Target Leakage / Redundant)
    leaky_columns = [
        "congestion_level",
        "traffic_stress_index",
        "peak_weight",
        "rain_norm",
        "wind_norm",
        "weather_severity",
        "time"
    ]

    return categorical_features, numerical_features, target_col, leaky_columns


def prepare_ml_dataset(df):
    """
    Takes a preprocessed dataframe, computes derived features, and extracts
    the feature matrix X and target vector y with zero target leakage.
    """
    df_engineered = add_derived_features(df)
    
    cat_cols, num_cols, target_col, leaky_cols = get_feature_lists()
    
    feature_cols = cat_cols + num_cols
    
    # Verify all required feature columns exist
    missing_cols = [col for col in feature_cols if col not in df_engineered.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns in dataset: {missing_cols}")

    X = df_engineered[feature_cols].copy()
    y = df_engineered[target_col].copy() if target_col in df_engineered.columns else None

    print(f"[INFO] Feature matrix X shape: {X.shape} ({len(cat_cols)} categorical, {len(num_cols)} numerical)")
    if y is not None:
        print(f"[INFO] Target vector y shape: {y.shape}")
        
    return X, y, df_engineered


if __name__ == "__main__":
    from preprocessing import preprocess_data
    df_clean = preprocess_data()
    X, y, df_eng = prepare_ml_dataset(df_clean)
    print("\nFeature matrix X head:")
    print(X.head(2))
    print("\nTarget y head:")
    print(y.head(2))
