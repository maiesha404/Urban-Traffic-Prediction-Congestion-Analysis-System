import os
import pandas as pd
import numpy as np

COLUMN_MAPPING = {
    "time": "time",
    "Area Name": "area_name",
    "Road/Intersection Name": "road_name",
    "Traffic Volume": "traffic_volume",
    "Average Speed": "average_speed",
    "Travel Time Index": "travel_time_index",
    "Congestion Level": "congestion_level",
    "Road Capacity Utilization": "road_capacity_utilization",
    "Incident Reports": "incident_reports",
    "Environmental Impact": "environmental_impact",
    "Public Transport Usage": "public_transport_usage",
    "Traffic Signal Compliance": "traffic_signal_compliance",
    "Parking Usage": "parking_usage",
    "Pedestrian and Cyclist Count": "pedestrian_cyclist_count",
    "Weather Conditions": "weather_conditions",
    "Roadwork and Construction Activity": "roadwork_activity",
    "temp_min": "temp_min",
    "temp_max": "temp_max",
    "temp_mean": "temp_mean",
    "rain_mm": "rain_mm",
    "wind_speed": "wind_speed",
    "weekday": "weekday",
    "is_peak": "is_peak",
    "peak_weight": "peak_weight",
    "rain_norm": "rain_norm",
    "wind_norm": "wind_norm",
    "weather_severity": "weather_severity",
    "traffic_stress_index": "traffic_stress_index"
}


def get_data_path(filename="bengaluru_traffic_weather_final.csv"):
    
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_file_dir) if os.path.basename(current_file_dir) == "src" else current_file_dir
    
    possible_paths = [
        os.path.join(project_root, "data", filename),
        os.path.join("data", filename),
        os.path.join("..", "data", filename),
        os.path.join(project_root, filename),
        filename,
        os.path.join("..", filename)
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    raise FileNotFoundError(f"Could not locate dataset file: {filename}")

def load_raw_dataset(filepath=None):
    """
    Loads the main traffic-weather dataset from CSV.
    """
    if filepath is None:
        filepath = get_data_path("bengaluru_traffic_weather_final.csv")
    
    print(f"[INFO] Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    return df


def verify_and_clean_data(df):
    """
    Performs data quality checks:
    - Missing value count
    - Duplicate row check
    - Domain bounds validation (positive volume/speed, congestion in [0, 100])
    """
    print("\n--- Running Data Health & Validation Checks ---")
    
    # 1. Missing values check
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    print(f"[CHECK] Total Missing Values: {total_missing}")
    if total_missing > 0:
        print("[WARN] Columns with missing values:\n", missing_counts[missing_counts > 0])
        # Drop rows with nulls if any
        df = df.dropna().reset_index(drop=True)
    else:
        print("[PASS] No missing values detected.")

    # 2. Duplicate rows check
    duplicates = df.duplicated().sum()
    print(f"[CHECK] Duplicate Rows: {duplicates}")
    if duplicates > 0:
        print(f"[WARN] Removing {duplicates} duplicate entries.")
        df = df.drop_duplicates().reset_index(drop=True)
    else:
        print("[PASS] Zero duplicate records found.")

    # 3. Value range & domain constraint validation
    # Standardize names temporarily for validation if needed
    vol_col = "Traffic Volume" if "Traffic Volume" in df.columns else "traffic_volume"
    speed_col = "Average Speed" if "Average Speed" in df.columns else "average_speed"
    cong_col = "Congestion Level" if "Congestion Level" in df.columns else "congestion_level"

    if vol_col in df.columns:
        invalid_vols = (df[vol_col] < 0).sum()
        print(f"[CHECK] Invalid Traffic Volume (< 0): {invalid_vols}")
        if invalid_vols > 0:
            df = df[df[vol_col] >= 0]

    if speed_col in df.columns:
        invalid_speeds = (df[speed_col] < 0).sum()
        print(f"[CHECK] Invalid Average Speed (< 0): {invalid_speeds}")
        if invalid_speeds > 0:
            df = df[df[speed_col] >= 0]

    if cong_col in df.columns:
        invalid_cong = ((df[cong_col] < 0) | (df[cong_col] > 100)).sum()
        print(f"[CHECK] Congestion Level out-of-bounds ([0, 100]): {invalid_cong}")
        if invalid_cong > 0:
            df = df[(df[cong_col] >= 0) & (df[cong_col] <= 100)]

    print("[PASS] Domain validity checks passed successfully.\n")
    return df


def preprocess_data(filepath=None):
    """
    Main preprocessing pipeline:
    1. Loads dataset
    2. Parses datetime
    3. Runs health checks
    4. Renames columns to clean snake_case
    """
    df = load_raw_dataset(filepath)

    # 1. Parse 'time' into datetime format
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    elif "Date" in df.columns:
        df["time"] = pd.to_datetime(df["Date"])

    # 2. Run data validation checks
    df = verify_and_clean_data(df)

    # 3. Standardize column names
    df = df.rename(columns=COLUMN_MAPPING)

    # Strip any residual whitespace from string columns
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip()

    print(f"[SUCCESS] Preprocessed dataframe shape: {df.shape}")
    return df


def demonstrate_raw_merge():
    """
    Educational helper for college report:
    Demonstrates how raw traffic data and raw weather data are merged on date.
    """
    print("\n--- Demonstrating Raw Merge of Traffic + Weather Datasets ---")
    try:
        traffic_path = get_data_path("banglore_traffic.csv")
        weather_path = get_data_path("weather_2020_2025.csv")
        
        df_traffic = pd.read_csv(traffic_path)
        df_weather = pd.read_csv(weather_path)
        
        # Convert date columns to common date format
        df_traffic["Date"] = pd.to_datetime(df_traffic["Date"])
        df_weather["time"] = pd.to_datetime(df_weather["time"])
        
        merged_df = pd.merge(
            df_traffic, 
            df_weather, 
            left_on="Date", 
            right_on="time", 
            how="inner"
        )
        print(f"[INFO] Raw Traffic rows: {len(df_traffic)}, Raw Weather rows: {len(df_weather)}")
        print(f"[INFO] Successfully merged inner join rows: {len(merged_df)}")
        return merged_df
    except Exception as e:
        print(f"[WARN] Raw merge demo skipped: {e}")
        return None


if __name__ == "__main__":
    df_clean = preprocess_data()
    print("Cleaned DataFrame sample preview:")
    print(df_clean.head(3))
    demonstrate_raw_merge()
