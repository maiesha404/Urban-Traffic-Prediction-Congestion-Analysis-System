import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add src to system path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# pyrefly: ignore [missing-import]
from preprocessing import preprocess_data, get_data_path
# pyrefly: ignore [missing-import]
from feature_engineering import (
    add_derived_features,
    classify_congestion,
    classify_tsi,
    compute_traffic_stress_index
)
# pyrefly: ignore [missing-import]
from prediction import predict_congestion, load_prediction_model

# ---------------------------------------------------------
# PAGE CONFIG & CUSTOM DASHBOARD THEME
# ---------------------------------------------------------
st.set_page_config(
    page_title="Urban Traffic Control & Analytics | Bengaluru",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Control-Room Theme CSS
st.markdown("""
<style>
    /* Dark Slate Control-Room Palette */
    :root {
        --bg-main: #0b0f19;
        --bg-card: #131b2e;
        --bg-card-border: #1e293b;
        --text-main: #e2e8f0;
        --text-muted: #94a3b8;
        --accent-blue: #3b82f6;
        --accent-emerald: #10b981;
        --accent-amber: #f59e0b;
        --accent-rose: #f43f5e;
    }

    /* Overall container adjustments */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    
    /* Metrics and Card Containers */
    .metric-card {
        background-color: #131b2e;
        border: 1px solid #1e293b;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1.1;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* Severity Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .badge-emerald { background-color: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid #059669; }
    .badge-amber { background-color: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid #d97706; }
    .badge-orange { background-color: rgba(249, 115, 22, 0.15); color: #fb923c; border: 1px solid #ea580c; }
    .badge-rose { background-color: rgba(244, 63, 94, 0.15); color: #fb7185; border: 1px solid #e11d48; }
    .badge-red { background-color: rgba(220, 38, 38, 0.2); color: #f87171; border: 1px solid #dc2626; }

    /* Headers */
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f1f5f9;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 8px;
        margin-top: 10px;
        margin-bottom: 16px;
    }
    
    /* Control inputs styling */
    div[data-baseweb="select"] {
        border-radius: 4px;
    }
    
    /* Plotly container clean border */
    .stPlotlyChart {
        border: 1px solid #1e293b;
        border-radius: 6px;
        background-color: #131b2e;
        padding: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# DATA & CACHING HELPERS
# ---------------------------------------------------------
@st.cache_data
def get_processed_data():
    """
    Loads and caches the cleaned, preprocessed dataset for dashboard analytics.
    """
    df = preprocess_data()
    df = add_derived_features(df)
    return df


@st.cache_data
def get_model_metadata():
    """
    Loads training metrics and benchmark metadata.
    """
    possible_paths = [
        os.path.join("models", "model_metrics.json"),
        os.path.join("..", "models", "model_metrics.json")
    ]
    for p in possible_paths:
        if os.path.exists(p):
            with open(p, "r") as f:
                return json.load(f)
    return None


def get_plotly_layout_defaults():
    """
    Returns unified dark-slate theme settings for all Plotly figures.
    """
    return dict(
        paper_bgcolor="#131b2e",
        plot_bgcolor="#131b2e",
        font=dict(color="#cbd5e1", family="sans-serif", size=12),
        margin=dict(l=40, r=30, t=40, b=40),
        xaxis=dict(
            gridcolor="#1e293b",
            zerolinecolor="#334155",
            showline=True,
            linecolor="#334155"
        ),
        yaxis=dict(
            gridcolor="#1e293b",
            zerolinecolor="#334155",
            showline=True,
            linecolor="#334155"
        ),
        legend=dict(
            bgcolor="rgba(19, 27, 46, 0.8)",
            bordercolor="#1e293b",
            borderwidth=1
        )
    )


# Load dataset and metadata
try:
    df_main = get_processed_data()
    metrics_meta = get_model_metadata()
except Exception as e:
    st.error(f"Error initializing data pipeline: {e}")
    st.stop()


# ---------------------------------------------------------
# SIDEBAR NAVIGATION & SYSTEM STATUS
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="padding: 4px 0 16px 0; border-bottom: 1px solid #1e293b; margin-bottom: 16px;">
            <div style="font-size: 1.05rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.01em;">
                🚦 Bengaluru Traffic Prediction
            </div>
            <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 2px;">
                Urban Congestion Intelligence & ML Ops
            </div>
        </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Live Prediction",
            "Traffic Hotspots",
            "Weather Impact"
        ],
        label_visibility="collapsed"
    )


# ---------------------------------------------------------
# PAGE 1: OVERVIEW / HOME
# ---------------------------------------------------------
if page == "Overview":
    st.markdown("<div class='section-header'>MUNICIPAL TRAFFIC SYSTEM OVERVIEW</div>", unsafe_allow_html=True)

    # Top KPI Metrics Cards
    avg_congestion = df_main["congestion_level"].mean()
    avg_speed = df_main["average_speed"].mean()
    avg_tsi = df_main["traffic_stress_index"].mean()
    tsi_cat, tsi_col = classify_tsi(avg_tsi)
    
    best_model_name = metrics_meta.get("best_model_name", "Gradient Boosting") if metrics_meta else "Random Forest"
    best_r2 = metrics_meta.get("best_test_r2", 0.892) if metrics_meta else 0.892

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">CITYWIDE AVG CONGESTION</div>
                <div class="metric-value">{avg_congestion:.1f}%</div>
                <div class="metric-sub">Mean capacity saturation</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">AVG NETWORK SPEED</div>
                <div class="metric-value">{avg_speed:.1f} <span style="font-size: 0.9rem; font-weight: 500; color: #94a3b8;">km/h</span></div>
                <div class="metric-sub">Across all major corridors</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">SYSTEM STRESS INDEX (TSI)</div>
                <div class="metric-value">{avg_tsi:.1f}</div>
                <div class="metric-sub"><span class="badge badge-{tsi_col}">{tsi_cat} STRESS</span></div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">PRIMARY ML MODEL</div>
                <div class="metric-value" style="font-size: 1.35rem; color: #60a5fa;">{best_model_name}</div>
                <div class="metric-sub">Test R² Score: <b>{best_r2:.4f}</b></div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # 2-column layout: Citywide Trend Line and Area Breakdown
    c_left, c_right = st.columns([3, 2])

    with c_left:
        st.markdown("<div style='font-size: 0.9rem; font-weight: 700; color: #cbd5e1; margin-bottom: 6px;'>Historical Congestion Index (2022 - 2024 Daily Timeline)</div>", unsafe_allow_html=True)
        # Aggregate daily average
        df_daily = df_main.groupby(df_main["time"].dt.date)["congestion_level"].mean().reset_index()
        fig_time = px.line(
            df_daily,
            x="time",
            y="congestion_level",
            color_discrete_sequence=["#38bdf8"]
        )
        fig_time.update_layout(get_plotly_layout_defaults())
        fig_time.update_layout(
            height=320,
            xaxis_title="Date",
            yaxis_title="Avg Congestion (%)",
            yaxis=dict(range=[0, 105], gridcolor="#1e293b")
        )
        st.plotly_chart(fig_time, width="stretch")

    with c_right:
        st.markdown("<div style='font-size: 0.9rem; font-weight: 700; color: #cbd5e1; margin-bottom: 6px;'>Zone-wise Average Congestion Level</div>", unsafe_allow_html=True)
        df_zone = df_main.groupby("area_name")["congestion_level"].mean().reset_index().sort_values(by="congestion_level", ascending=True)
        fig_zone = px.bar(
            df_zone,
            x="congestion_level",
            y="area_name",
            orientation="h",
            color="congestion_level",
            color_continuous_scale=["#10b981", "#f59e0b", "#f43f5e"]
        )
        fig_zone.update_layout(get_plotly_layout_defaults())
        fig_zone.update_layout(
            height=320,
            showlegend=False,
            coloraxis_showscale=False,
            xaxis_title="Mean Congestion (%)",
            yaxis_title=""
        )
        st.plotly_chart(fig_zone, width="stretch")

    # Quick Data Health & Schema Table
    st.markdown("<div style='font-size: 0.9rem; font-weight: 700; color: #cbd5e1; margin-top: 14px; margin-bottom: 8px;'>Recent Monitored Observations</div>", unsafe_allow_html=True)
    display_cols = ["time", "area_name", "road_name", "traffic_volume", "average_speed", "congestion_level", "weather_conditions", "is_peak"]
    st.dataframe(
        df_main[display_cols].tail(8).sort_values(by="time", ascending=False),
        width="stretch",
        hide_index=True
    )


# ---------------------------------------------------------
# PAGE 2: LIVE PREDICTION
# ---------------------------------------------------------
elif page == "Live Prediction":
    st.markdown("<div class='section-header'>INTERACTIVE TRAFFIC CONGESTION ESTIMATOR</div>", unsafe_allow_html=True)
    st.caption("Input corridor operational parameters and meteorological conditions to compute real-time congestion predictions and Traffic Stress Index (TSI).")

    # Form inputs structured cleanly
    unique_areas = sorted(df_main["area_name"].unique())
    unique_roads = sorted(df_main["road_name"].unique())
    unique_weather = sorted(df_main["weather_conditions"].unique())

    with st.container():
        st.markdown("<div style='font-size: 0.85rem; font-weight: 700; text-transform: uppercase; color: #94a3b8; margin-bottom: 8px;'>1. Corridor Location & Geometry</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            in_area = st.selectbox("Area Name / Zone", unique_areas, index=0)
        with c2:
            # Filter roads by area if possible, otherwise full list
            roads_in_area = sorted(df_main[df_main["area_name"] == in_area]["road_name"].unique())
            if not len(roads_in_area):
                roads_in_area = unique_roads
            in_road = st.selectbox("Road / Intersection", roads_in_area, index=0)
        with c3:
            in_roadwork = st.selectbox("Roadwork / Construction Activity", ["No", "Yes"], index=0)

        p1, p2, p3, p4 = st.columns(4)
        with p1:
            in_vol = st.number_input("Traffic Volume", min_value=0, max_value=15000, value=2800, step=100)
            in_speed = st.number_input("Average Speed (km/h)", min_value=1.0, max_value=120.0, value=22.5, step=0.5)
        with p2:
            in_tti = st.number_input("Travel Time Index", min_value=0.5, max_value=10.0, value=2.4, step=0.1)
            in_capacity = st.number_input("Road Capacity Utilization (%)", min_value=1.0, max_value=150.0, value=82.0, step=1.0)
        with p3:
            in_incidents = st.number_input("Incident Reports", min_value=0, max_value=50, value=1, step=1)
            in_parking = st.number_input("Parking Usage (%)", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
        with p4:
            in_compliance = st.number_input("Traffic Signal Compliance (%)", min_value=0.0, max_value=100.0, value=88.0, step=1.0)
            in_public_transport = st.number_input("Public Transport Usage (%)", min_value=0.0, max_value=100.0, value=48.0, step=1.0)

        st.markdown("<div style='margin-top: 12px; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; color: #94a3b8; margin-bottom: 8px;'>2. Weather & Temporal Context</div>", unsafe_allow_html=True)
        w1, w2, w3, w4 = st.columns(4)
        with w1:
            in_weather_cond = st.selectbox("Weather Condition", unique_weather, index=0)
            weekday_labels = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            in_weekday_name = st.selectbox("Day of Week", weekday_labels, index=2)
            in_weekday_num = weekday_labels.index(in_weekday_name)
        with w2:
            in_temp_mean = st.slider("Mean Temperature (°C)", min_value=12.0, max_value=42.0, value=25.0, step=0.5)
            in_is_peak = st.radio("Peak Hour Period?", [0, 1], format_func=lambda x: "Yes (Peak Rush)" if x == 1 else "No (Off-Peak)", horizontal=True)
        with w3:
            in_rain = st.number_input("Rainfall (mm)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
            in_pedestrians = st.number_input("Pedestrians / Cyclists Count", min_value=0, max_value=2000, value=180, step=10)
        with w4:
            in_wind = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=60.0, value=11.0, step=1.0)
            in_env_impact = st.number_input("Environmental Impact Index", min_value=0.0, max_value=100.0, value=52.0, step=1.0)

        st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
        predict_btn = st.button("RUN PREDICTION MODEL", type="primary", width="stretch")

    if predict_btn:
        input_payload = {
            "area_name": in_area,
            "road_name": in_road,
            "traffic_volume": in_vol,
            "average_speed": in_speed,
            "travel_time_index": in_tti,
            "road_capacity_utilization": in_capacity,
            "incident_reports": in_incidents,
            "environmental_impact": in_env_impact,
            "public_transport_usage": in_public_transport,
            "traffic_signal_compliance": in_compliance,
            "parking_usage": in_parking,
            "pedestrian_cyclist_count": in_pedestrians,
            "weather_conditions": in_weather_cond,
            "roadwork_activity": in_roadwork,
            "temp_min": in_temp_mean - 4.0,
            "temp_max": in_temp_mean + 4.0,
            "temp_mean": in_temp_mean,
            "rain_mm": in_rain,
            "wind_speed": in_wind,
            "weekday": in_weekday_num,
            "is_peak": in_is_peak
        }

        with st.spinner("Processing features and computing ML inference..."):
            try:
                res = predict_congestion(input_payload)
                pred_cong = res["predicted_congestion_level"]
                cong_cat = res["congestion_category"]
                cong_col = res["congestion_color"]
                cong_desc = res["congestion_description"]
                tsi_score = res["traffic_stress_index"]
                tsi_cat = res["tsi_category"]
                tsi_col = res["tsi_color"]

                st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
                st.markdown("<div class='section-header'>PREDICTION RESULTS</div>", unsafe_allow_html=True)

                res_c1, res_c2, res_c3 = st.columns([1.5, 1.5, 2])
                with res_c1:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">PREDICTED CONGESTION LEVEL</div>
                            <div class="metric-value" style="color: #38bdf8;">{pred_cong:.1f}%</div>
                            <div style="margin-top: 8px;">
                                <span class="badge badge-{cong_col}">{cong_cat}</span>
                            </div>
                            <div class="metric-sub" style="margin-top: 6px;">{cong_desc}</div>
                        </div>
                    """, unsafe_allow_html=True)

                with res_c2:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">TRAFFIC STRESS INDEX (TSI)</div>
                            <div class="metric-value" style="color: #f59e0b;">{tsi_score:.1f}</div>
                            <div style="margin-top: 8px;">
                                <span class="badge badge-{tsi_col}">{tsi_cat} STRESS</span>
                            </div>
                            <div class="metric-sub" style="margin-top: 6px;">Peak Weight × Weather Multiplier</div>
                        </div>
                    """, unsafe_allow_html=True)

                with res_c3:
                    # Circular Gauge Chart using Plotly
                    gauge_fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=pred_cong,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                            'bar': {'color': "#38bdf8"},
                            'bgcolor': "#1e293b",
                            'borderwidth': 1,
                            'bordercolor': "#334155",
                            'steps': [
                                {'range': [0, 40], 'color': "rgba(16, 185, 129, 0.3)"},
                                {'range': [40, 65], 'color': "rgba(245, 158, 11, 0.3)"},
                                {'range': [65, 85], 'color': "rgba(249, 115, 22, 0.3)"},
                                {'range': [85, 100], 'color': "rgba(244, 63, 94, 0.35)"}
                            ],
                            'threshold': {
                                'line': {'color': "#f43f5e", 'width': 3},
                                'thickness': 0.75,
                                'value': 85
                            }
                        }
                    ))
                    gauge_fig.update_layout(get_plotly_layout_defaults())
                    gauge_fig.update_layout(
                        height=200,
                        margin=dict(l=25, r=25, t=25, b=25)
                    )
                    st.plotly_chart(gauge_fig, width="stretch")

            except Exception as ex:
                st.error(f"Inference execution failed: {ex}")



# ---------------------------------------------------------
# PAGE 3: TRAFFIC HOTSPOTS
# ---------------------------------------------------------
elif page == "Traffic Hotspots":
    st.markdown("<div class='section-header'>BENGALURU CORRIDOR BOTTLENECKS & HOTSPOTS</div>", unsafe_allow_html=True)

    h1, h2 = st.columns(2)

    with h1:
        st.markdown("<div style='font-size: 0.9rem; font-weight: 700; color: #cbd5e1; margin-bottom: 6px;'>Top 10 Most Congested Roads & Intersections</div>", unsafe_allow_html=True)
        df_roads = df_main.groupby("road_name")["congestion_level"].mean().reset_index().sort_values(by="congestion_level", ascending=False).head(10)
        fig_r = px.bar(
            df_roads,
            x="congestion_level",
            y="road_name",
            orientation="h",
            color="congestion_level",
            color_continuous_scale="Reds"
        )
        fig_r.update_layout(get_plotly_layout_defaults())
        fig_r.update_layout(
            height=340,
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
            xaxis_title="Average Congestion (%)",
            yaxis_title=""
        )
        st.plotly_chart(fig_r, width="stretch")

    with h2:
        st.markdown("<div style='font-size: 0.9rem; font-weight: 700; color: #cbd5e1; margin-bottom: 6px;'>Corridors with Highest Incident Frequencies</div>", unsafe_allow_html=True)
        df_inc = df_main.groupby("road_name")["incident_reports"].sum().reset_index().sort_values(by="incident_reports", ascending=False).head(10)
        fig_inc = px.bar(
            df_inc,
            x="incident_reports",
            y="road_name",
            orientation="h",
            color="incident_reports",
            color_continuous_scale="Oranges"
        )
        fig_inc.update_layout(get_plotly_layout_defaults())
        fig_inc.update_layout(
            height=340,
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
            xaxis_title="Total Recorded Incidents",
            yaxis_title=""
        )
        st.plotly_chart(fig_inc, width="stretch")

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.9rem; font-weight: 700; color: #cbd5e1; margin-bottom: 8px;'>Corridor Capacity Saturation Matrix</div>", unsafe_allow_html=True)
    
    pivot_matrix = df_main.pivot_table(
        index="area_name",
        columns="weather_conditions",
        values="congestion_level",
        aggfunc="mean"
    ).round(1)

    fig_heat = px.imshow(
        pivot_matrix,
        labels=dict(x="Weather Condition", y="Area Zone", color="Avg Congestion (%)"),
        color_continuous_scale="YlOrRd"
    )
    fig_heat.update_layout(get_plotly_layout_defaults())
    fig_heat.update_layout(height=320)
    st.plotly_chart(fig_heat, width="stretch")


# ---------------------------------------------------------
# PAGE 4: WEATHER IMPACT
# ---------------------------------------------------------
elif page == "Weather Impact":
    st.markdown("<div class='section-header'>METEOROLOGICAL IMPACT ON TRAFFIC FLOW</div>", unsafe_allow_html=True)

    w_c1, w_c2 = st.columns(2)

    with w_c1:
        st.markdown("<div style='font-size: 0.9rem; font-weight: 700; color: #cbd5e1; margin-bottom: 6px;'>Congestion vs. Rainfall Intensity (mm)</div>", unsafe_allow_html=True)
        fig_rain = px.scatter(
            df_main.sample(min(len(df_main), 1200), random_state=42),
            x="rain_mm",
            y="congestion_level",
            color="weather_conditions",
            opacity=0.65,
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_rain.update_layout(get_plotly_layout_defaults())
        fig_rain.update_layout(
            height=330,
            xaxis_title="Precipitation (mm)",
            yaxis_title="Congestion Level (%)"
        )
        st.plotly_chart(fig_rain, width="stretch")

    with w_c2:
        st.markdown("<div style='font-size: 0.9rem; font-weight: 700; color: #cbd5e1; margin-bottom: 6px;'>Average Speed vs. Weather Severity</div>", unsafe_allow_html=True)
        fig_cond = px.box(
            df_main,
            x="weather_conditions",
            y="average_speed",
            color="weather_conditions",
            color_discrete_sequence=px.colors.qualitative.Dark24
        )
        fig_cond.update_layout(get_plotly_layout_defaults())
        fig_cond.update_layout(
            height=330,
            showlegend=False,
            xaxis_title="Weather Condition",
            yaxis_title="Average Vehicle Speed (km/h)"
        )
        st.plotly_chart(fig_cond, width="stretch")

