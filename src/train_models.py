"""
Urban Traffic Prediction & Congestion Analysis System
Module: Model Training & Evaluation (train_models.py)

This module handles:
1. Loading and sorting preprocessed data chronologically.
2. 80/20 Chronological Train/Test Split (ensuring no temporal leakage).
3. Preprocessing Pipeline: OneHotEncoder for categoricals + StandardScaler for numericals.
4. Training and evaluating 4 regression algorithms:
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
   - Gradient Boosting Regressor
5. Metric evaluation: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R² Score.
6. Hyperparameter tuning of the top performer with GridSearchCV.
7. Diagnostic Visualizations saved to outputs/ (Comparison, Actual vs Predicted, Residuals, Feature Importance).
8. Serializing the final production pipeline to models/best_model.pkl and metrics to models/model_metrics.json.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

# Add src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from preprocessing import preprocess_data
from feature_engineering import prepare_ml_dataset, get_feature_lists


def build_preprocessor(categorical_cols, numerical_cols):
    """
    Constructs a ColumnTransformer to handle one-hot encoding of categoricals
    and standard scaling of numerical features.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols
            ),
            (
                "num",
                StandardScaler(),
                numerical_cols
            )
        ],
        remainder="drop"
    )
    return preprocessor


def evaluate_model(model, X_train, y_train, X_test, y_test, name="Model"):
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Clip predictions to sensible [0, 100] congestion percentage bounds
    y_pred_train = np.clip(y_pred_train, 0.0, 100.0)
    y_pred_test = np.clip(y_pred_test, 0.0, 100.0)

    # Calculate metrics
    mae_train = mean_absolute_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train = r2_score(y_train, y_pred_train)

    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)

    return {
        "name": name,
        "model": model,
        "train_mae": float(mae_train),
        "train_rmse": float(rmse_train),
        "train_r2": float(r2_train),
        "test_mae": float(mae_test),
        "test_rmse": float(rmse_test),
        "test_r2": float(r2_test),
        "y_pred_test": y_pred_test
    }


def plot_model_comparison(results, output_dir="outputs"):
    """
    Generates a grouped bar chart comparing MAE, RMSE, and R2 across all 4 models.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    names = [r["name"] for r in results]
    maes = [r["test_mae"] for r in results]
    rmses = [r["test_rmse"] for r in results]
    r2s = [r["test_r2"] for r in results]

    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    x = np.arange(len(names))
    width = 0.25

    rects1 = ax1.bar(x - width, maes, width, label="MAE (Test)", color="#3b82f6")
    rects2 = ax1.bar(x, rmses, width, label="RMSE (Test)", color="#f97316")

    ax1.set_ylabel("Error (MAE / RMSE)", fontsize=11, fontweight="bold", color="#1e293b")
    ax1.set_title("Model Performance Benchmark on Test Set (80/20 Chronological Split)", fontsize=13, fontweight="bold", pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=10, fontweight="bold")
    ax1.set_ylim(0, max(rmses) * 1.25)

    # Secondary axis for R2 score
    ax2 = ax1.twinx()
    rects3 = ax2.bar(x + width, r2s, width, label="R² Score (Test)", color="#10b981")
    ax2.set_ylabel("R² Score", fontsize=11, fontweight="bold", color="#10b981")
    ax2.set_ylim(0, 1.05)
    ax2.grid(False)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True)

    # Add text labels on top of bars
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for rect in rects3:
        h = rect.get_height()
        ax2.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8, color="#047857", fontweight="bold")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[SAVED] Model comparison chart -> {plot_path}", flush=True)


def plot_actual_vs_predicted(y_test, y_pred, model_name, output_dir="outputs"):
    """
    Generates Actual vs Predicted scatter plot with identity diagonal line.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    
    ax.scatter(y_test, y_pred, alpha=0.45, color="#2563eb", edgecolors="none", s=25, label="Test Predictions")
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], color="#ef4444", linestyle="--", linewidth=2, label="Ideal Fit (y = x)")

    ax.set_xlabel("Actual Congestion Level (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Predicted Congestion Level (%)", fontsize=11, fontweight="bold")
    ax.set_title(f"Actual vs. Predicted Congestion ({model_name})", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=True, loc="upper left")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "actual_vs_predicted.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[SAVED] Actual vs. Predicted chart -> {plot_path}", flush=True)


def plot_residuals(y_test, y_pred, model_name, output_dir="outputs"):
    """
    Generates residual diagnostics: Scatter of residuals vs predicted & residual histogram.
    """
    os.makedirs(output_dir, exist_ok=True)
    residuals = y_test - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # 1. Residuals vs Predicted
    ax1.scatter(y_pred, residuals, alpha=0.45, color="#6366f1", s=22, edgecolors="none")
    ax1.axhline(0, color="#ef4444", linestyle="--", linewidth=1.8)
    ax1.set_xlabel("Predicted Congestion Level (%)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Residual (Actual - Predicted)", fontsize=10, fontweight="bold")
    ax1.set_title("Residuals vs. Predicted Values", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # 2. Residual Distribution Histogram
    sns.histplot(residuals, kde=True, ax=ax2, color="#0ea5e9", bins=30, edgecolor="white")
    ax2.axvline(0, color="#ef4444", linestyle="--", linewidth=1.8)
    ax2.set_xlabel("Residual Error", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Frequency", fontsize=10, fontweight="bold")
    ax2.set_title("Residual Error Distribution", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle(f"Residual Diagnostics ({model_name})", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "residual_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[SAVED] Residual diagnostics chart -> {plot_path}", flush=True)


def plot_feature_importance(pipeline, cat_cols, num_cols, model_name, output_dir="outputs"):
    """
    Extracts and visualizes feature importances from tree-based estimators.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    regressor = pipeline.named_steps["regressor"]
    preprocessor = pipeline.named_steps["preprocessor"]

    cat_encoder = preprocessor.named_transformers_["cat"]
    encoded_cat_names = list(cat_encoder.get_feature_names_out(cat_cols))
    all_feature_names = encoded_cat_names + num_cols

    if hasattr(regressor, "feature_importances_"):
        importances = regressor.feature_importances_
    elif hasattr(regressor, "coef_"):
        importances = np.abs(regressor.coef_)
    else:
        print("[WARN] Estimator does not support feature importance extraction.", flush=True)
        return

    feat_df = pd.DataFrame({
        "feature": all_feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    sns.barplot(data=feat_df, x="importance", y="feature", palette="Blues_r", ax=ax)
    
    ax.set_xlabel("Relative Importance Score", fontsize=11, fontweight="bold")
    ax.set_ylabel("Engineered Feature", fontsize=11, fontweight="bold")
    ax.set_title(f"Top 15 Feature Importances ({model_name})", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[SAVED] Feature importance chart -> {plot_path}", flush=True)


def train_and_evaluate_all():
    """
    Main training execution pipeline.
    """
    print("\n=======================================================", flush=True)
    print(" URBAN TRAFFIC PREDICTION & CONGESTION MODEL TRAINING ", flush=True)
    print("=======================================================\n", flush=True)

    # 1. Load preprocessed data
    df = preprocess_data()

    # 2. Sort chronologically by time
    df = df.sort_values(by="time").reset_index(drop=True)
    print(f"[INFO] Chronological range: {df['time'].min().date()} to {df['time'].max().date()}", flush=True)

    # 3. Extract feature matrix X and target y with zero target leakage
    X, y, df_engineered = prepare_ml_dataset(df)
    cat_cols, num_cols, target_col, _ = get_feature_lists()

    # 4. Chronological 80/20 train/test split (no shuffling)
    split_idx = int(len(df) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"[INFO] Train Set: {len(X_train)} samples ({df['time'].iloc[0].date()} -> {df['time'].iloc[split_idx-1].date()})", flush=True)
    print(f"[INFO] Test Set:  {len(X_test)} samples ({df['time'].iloc[split_idx].date()} -> {df['time'].iloc[-1].date()})", flush=True)

    # 5. Define candidate models
    preprocessor = build_preprocessor(cat_cols, num_cols)

    candidate_models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=10, min_samples_split=10),
        "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100, max_depth=14, min_samples_split=6),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42, n_estimators=100, max_depth=5, learning_rate=0.08)
    }

    results = []
    print("\n--- Training & Evaluating 4 ML Algorithms ---", flush=True)
    for name, reg in candidate_models.items():
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", reg)
        ])
        res = evaluate_model(pipe, X_train, y_train, X_test, y_test, name=name)
        results.append(res)
        print(f"-> {name:<20} | Test MAE: {res['test_mae']:.3f} | Test RMSE: {res['test_rmse']:.3f} | Test R²: {res['test_r2']:.4f}", flush=True)

    # Display comparison table
    summary_df = pd.DataFrame([{
        "Model": r["name"],
        "Train MAE": round(r["train_mae"], 3),
        "Test MAE": round(r["test_mae"], 3),
        "Train RMSE": round(r["train_rmse"], 3),
        "Test RMSE": round(r["test_rmse"], 3),
        "Train R²": round(r["train_r2"], 4),
        "Test R²": round(r["test_r2"], 4)
    } for r in results])
    
    print("\n--- Model Benchmark Summary Table ---", flush=True)
    print(summary_df.to_string(index=False), flush=True)

    # Identify best initial model by highest test R²
    best_init_result = max(results, key=lambda r: r["test_r2"])
    print(f"\n[INFO] Best Performing Baseline Model: {best_init_result['name']} (Test R² = {best_init_result['test_r2']:.4f})", flush=True)

    # 6. Hyperparameter Tuning with GridSearchCV
    print("\n--- Hyperparameter Tuning with GridSearchCV ---", flush=True)
    tune_target = best_init_result["name"]
    
    if "Random Forest" in tune_target:
        param_grid = {
            "regressor__n_estimators": [80, 120],
            "regressor__max_depth": [12, 16],
            "regressor__min_samples_split": [4, 8]
        }
        base_estimator = RandomForestRegressor(random_state=42)
    else:
        param_grid = {
            "regressor__n_estimators": [80, 120],
            "regressor__max_depth": [4, 6],
            "regressor__learning_rate": [0.05, 0.1]
        }
        base_estimator = GradientBoostingRegressor(random_state=42)

    tuning_pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", base_estimator)
    ])

    grid_search = GridSearchCV(
        tuning_pipe,
        param_grid=param_grid,
        cv=3,
        scoring="r2",
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    best_pipe = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"[INFO] Optimal Hyperparameters: {best_params}", flush=True)

    # Evaluate tuned model on test set
    tuned_eval = evaluate_model(best_pipe, X_train, y_train, X_test, y_test, name=f"Tuned {tune_target}")
    print(f"[TUNED RESULT] Test MAE: {tuned_eval['test_mae']:.3f} | Test RMSE: {tuned_eval['test_rmse']:.3f} | Test R²: {tuned_eval['test_r2']:.4f}", flush=True)

    # 7. Generate diagnostic plots
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_file_dir) if os.path.basename(current_file_dir) == "src" else current_file_dir

    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    plot_model_comparison(results, output_dir=output_dir)
    plot_actual_vs_predicted(y_test, tuned_eval["y_pred_test"], tuned_eval["name"], output_dir=output_dir)
    plot_residuals(y_test, tuned_eval["y_pred_test"], tuned_eval["name"], output_dir=output_dir)
    plot_feature_importance(best_pipe, cat_cols, num_cols, tuned_eval["name"], output_dir=output_dir)

    # 8. Save production pipeline & metadata
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, "best_model.pkl")

    joblib.dump(best_pipe, model_save_path)
    print(f"\n[SAVED] Serialized best model pipeline -> {model_save_path}", flush=True)

    # Save metrics JSON for Streamlit app consumption
    metrics_data = {
        "best_model_name": tuned_eval["name"],
        "best_test_r2": round(tuned_eval["test_r2"], 4),
        "best_test_mae": round(tuned_eval["test_mae"], 3),
        "best_test_rmse": round(tuned_eval["test_rmse"], 3),
        "best_params": best_params,
        "models_summary": summary_df.to_dict(orient="records"),
        "training_samples": len(X_train),
        "testing_samples": len(X_test),
        "feature_count": len(cat_cols) + len(num_cols),
        "categorical_columns": cat_cols,
        "numerical_columns": num_cols
    }

    metrics_save_path = os.path.join(models_dir, "model_metrics.json")
    with open(metrics_save_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"[SAVED] Exported model metrics metadata -> {metrics_save_path}", flush=True)


    print("\n=======================================================", flush=True)
    print(" MODEL TRAINING & ARTIFACT GENERATION COMPLETE ", flush=True)
    print("=======================================================\n", flush=True)
    return best_pipe, metrics_data


if __name__ == "__main__":
    train_and_evaluate_all()
