"""
Streamlit Dashboard for Metro Compressor Anomaly Detection

Real-time monitoring dashboard with:
- Live failure risk display
- Anomaly score visualization
- Alert system with threshold triggers
- Historical trend views
- Signal contribution analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path
from datetime import datetime, timedelta
import time

# Page config
st.set_page_config(
    page_title="Metro Compressor Monitor",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "dataset" / "metropt+3+dataset"
DATA_FILE = DATA_DIR / "MetroPT3(AirCompressor).csv"

# Constants
ANALOGUE_SIGNALS = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Oil_temperature', 'Motor_current']
DIGITAL_SIGNALS = ['COMP', 'DV_electric', 'Towers', 'MPG', 'LPS', 'Pressure_switch', 'Oil_level', 'Caudal_impulses']
WINDOW_SIZE = 60

# Thresholds (from training)
OIL_TEMP_THRESHOLD = 63.7  # mean + 2*std


@st.cache_resource
def load_model():
    """Load the trained Isolation Forest model."""
    model_path = MODEL_DIR / "isolation_forest.joblib"
    scaler_path = MODEL_DIR / "iforest_scaler.joblib"
    feature_cols_path = MODEL_DIR / "feature_cols.txt"

    if not model_path.exists():
        return None, None, None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(feature_cols_path) as f:
        feature_cols = [line.strip() for line in f if line.strip()]

    return model, scaler, feature_cols


@st.cache_data
def load_data():
    """Load the dataset."""
    if not DATA_FILE.exists():
        return None

    df = pd.read_csv(DATA_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def compute_rolling_features(window_df):
    """Compute rolling features from a window of data."""
    features = {}
    for signal in ANALOGUE_SIGNALS:
        data = window_df[signal].values
        features[signal] = data[-1]
        features[f'{signal}_mean'] = np.mean(data)
        features[f'{signal}_std'] = np.std(data)
        features[f'{signal}_min'] = np.min(data)
        features[f'{signal}_max'] = np.max(data)
    return features


def get_anomaly_score(features, model, scaler, feature_cols):
    """Get anomaly score from model."""
    feature_vector = np.array([features[col] for col in feature_cols]).reshape(1, -1)
    feature_scaled = scaler.transform(feature_vector)
    raw_score = model.score_samples(feature_scaled)[0]
    return -raw_score  # Convert to positive (higher = more anomalous)


def main():
    # Header
    st.title("🚇 Metro Compressor Anomaly Detection")
    st.markdown("Real-time monitoring for air compressor failure prediction")

    # Load resources
    model, scaler, feature_cols = load_model()
    df = load_data()

    if df is None:
        st.error("Dataset not found. Please ensure the data file exists.")
        return

    if model is None:
        st.warning("Model not loaded. Running in visualization-only mode.")

    # Sidebar controls
    st.sidebar.header("Controls")

    # Mode selection
    mode = st.sidebar.radio("Mode", ["Historical Analysis", "Live Simulation"])

    if mode == "Historical Analysis":
        historical_analysis(df, model, scaler, feature_cols)
    else:
        live_simulation(df, model, scaler, feature_cols)


def historical_analysis(df, model, scaler, feature_cols):
    """Historical data analysis mode."""
    st.header("Historical Analysis")

    # Failure period selection
    st.sidebar.subheader("Select Period")

    FAILURE_PERIODS = {
        "Failure #1 (Apr 13-14)": ("2020-04-13 06:00:00", "2020-04-14 18:00:00"),
        "Failure #2 (Apr 28-29)": ("2020-04-28 06:00:00", "2020-04-29 18:00:00"),
        "Failure #3 (Jun 23)": ("2020-06-23 06:00:00", "2020-06-23 18:00:00"),
        "Failure #4 (Aug 14-17)": ("2020-08-14 00:00:00", "2020-08-17 23:59:00"),
        "Custom Range": None
    }

    period = st.sidebar.selectbox("Failure Period", list(FAILURE_PERIODS.keys()))

    if period == "Custom Range":
        start_date = st.sidebar.date_input("Start Date", df['timestamp'].min().date())
        end_date = st.sidebar.date_input("End Date", df['timestamp'].max().date())
        start_time = datetime.combine(start_date, datetime.min.time())
        end_time = datetime.combine(end_date, datetime.max.time())
    else:
        start_time, end_time = FAILURE_PERIODS[period]
        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)

    # Filter data
    mask = (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)
    period_df = df[mask].copy()

    if len(period_df) == 0:
        st.warning("No data in selected period")
        return

    st.write(f"**Period:** {start_time} to {end_time} ({len(period_df):,} records)")

    # Compute anomaly scores if model available
    if model is not None:
        with st.spinner("Computing anomaly scores..."):
            scores = []
            for i in range(WINDOW_SIZE, len(period_df)):
                window = period_df.iloc[i-WINDOW_SIZE:i]
                features = compute_rolling_features(window)
                score = get_anomaly_score(features, model, scaler, feature_cols)
                scores.append(score)

            period_df = period_df.iloc[WINDOW_SIZE:].copy()
            period_df['anomaly_score'] = scores
            period_df['is_anomaly'] = period_df['anomaly_score'] > 0.5

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Records", f"{len(period_df):,}")

    with col2:
        if 'anomaly_score' in period_df.columns:
            anomaly_pct = 100 * period_df['is_anomaly'].mean()
            st.metric("Anomaly Rate", f"{anomaly_pct:.1f}%")

    with col3:
        max_oil = period_df['Oil_temperature'].max()
        delta = max_oil - OIL_TEMP_THRESHOLD
        st.metric("Max Oil Temp", f"{max_oil:.1f}°C", f"{delta:+.1f}°C" if delta > 0 else None)

    with col4:
        if 'anomaly_score' in period_df.columns:
            max_score = period_df['anomaly_score'].max()
            st.metric("Max Anomaly Score", f"{max_score:.3f}")

    # Signal plots
    st.subheader("Sensor Signals")

    signal_to_plot = st.multiselect(
        "Select signals to display",
        ANALOGUE_SIGNALS,
        default=['Oil_temperature', 'TP2', 'H1']
    )

    if signal_to_plot:
        fig = make_subplots(rows=len(signal_to_plot), cols=1, shared_xaxes=True,
                           subplot_titles=signal_to_plot, vertical_spacing=0.05)

        for i, signal in enumerate(signal_to_plot, 1):
            fig.add_trace(
                go.Scatter(x=period_df['timestamp'], y=period_df[signal],
                          name=signal, line=dict(width=1)),
                row=i, col=1
            )

            # Add threshold line for Oil_temperature
            if signal == 'Oil_temperature':
                fig.add_hline(y=OIL_TEMP_THRESHOLD, line_dash="dash",
                             line_color="red", row=i, col=1,
                             annotation_text="Threshold")

        fig.update_layout(height=200*len(signal_to_plot), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Anomaly score plot
    if 'anomaly_score' in period_df.columns:
        st.subheader("Anomaly Score Over Time")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=period_df['timestamp'],
            y=period_df['anomaly_score'],
            mode='lines',
            name='Anomaly Score',
            line=dict(color='purple', width=1)
        ))
        fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                     annotation_text="Anomaly Threshold")
        fig.update_layout(height=300, yaxis_title="Anomaly Score")
        st.plotly_chart(fig, use_container_width=True)

    # Digital signals
    with st.expander("Digital Signals"):
        fig = make_subplots(rows=len(DIGITAL_SIGNALS), cols=1, shared_xaxes=True,
                           subplot_titles=DIGITAL_SIGNALS, vertical_spacing=0.02)

        for i, signal in enumerate(DIGITAL_SIGNALS, 1):
            fig.add_trace(
                go.Scatter(x=period_df['timestamp'], y=period_df[signal],
                          name=signal, line=dict(width=1)),
                row=i, col=1
            )

        fig.update_layout(height=100*len(DIGITAL_SIGNALS), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def live_simulation(df, model, scaler, feature_cols):
    """Live simulation mode with streaming data."""
    st.header("Live Simulation")

    # Simulation controls
    col1, col2, col3 = st.sidebar.columns(3)

    FAILURE_STARTS = {
        "Failure #1": 200000,
        "Failure #2": 330000,
        "Failure #3": 1100000,
        "Failure #4": 1480000,
    }

    failure = st.sidebar.selectbox("Start from", list(FAILURE_STARTS.keys()))
    start_idx = FAILURE_STARTS[failure]

    speed = st.sidebar.slider("Speed (readings/sec)", 1, 100, 10)
    window_size = st.sidebar.slider("Display window (seconds)", 60, 600, 300)

    # Initialize session state
    if 'sim_running' not in st.session_state:
        st.session_state.sim_running = False
        st.session_state.sim_idx = start_idx
        st.session_state.history = []
        st.session_state.alerts = []

    # Control buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("▶️ Start" if not st.session_state.sim_running else "⏸️ Pause"):
            st.session_state.sim_running = not st.session_state.sim_running

    with col2:
        if st.button("🔄 Reset"):
            st.session_state.sim_running = False
            st.session_state.sim_idx = start_idx
            st.session_state.history = []
            st.session_state.alerts = []
            st.rerun()

    # Status indicators
    status_col1, status_col2, status_col3, status_col4 = st.columns(4)

    # Placeholders for live updates
    risk_placeholder = st.empty()
    chart_placeholder = st.empty()
    alerts_placeholder = st.empty()

    # Simulation loop
    if st.session_state.sim_running and st.session_state.sim_idx < len(df):
        # Process batch of readings
        batch_size = speed
        end_idx = min(st.session_state.sim_idx + batch_size, len(df))

        for idx in range(st.session_state.sim_idx, end_idx):
            row = df.iloc[idx]

            # Store in history
            record = {
                'timestamp': row['timestamp'],
                'Oil_temperature': row['Oil_temperature'],
                'TP2': row['TP2'],
                'H1': row['H1'],
                'Motor_current': row['Motor_current'],
            }

            # Compute anomaly score if we have enough history
            if len(st.session_state.history) >= WINDOW_SIZE and model is not None:
                window_df = pd.DataFrame(st.session_state.history[-WINDOW_SIZE:])
                features = compute_rolling_features(window_df)
                score = get_anomaly_score(features, model, scaler, feature_cols)
                record['anomaly_score'] = score
                record['is_anomaly'] = score > 0.5 or row['Oil_temperature'] > OIL_TEMP_THRESHOLD

                # Generate alert
                if record['is_anomaly']:
                    alert = f"⚠️ {row['timestamp']}: Anomaly detected (score: {score:.3f})"
                    if alert not in st.session_state.alerts[-10:]:
                        st.session_state.alerts.append(alert)
            else:
                record['anomaly_score'] = 0
                record['is_anomaly'] = False

            st.session_state.history.append(record)

        st.session_state.sim_idx = end_idx

        # Trim history to window size
        if len(st.session_state.history) > window_size:
            st.session_state.history = st.session_state.history[-window_size:]

        # Update displays
        if st.session_state.history:
            latest = st.session_state.history[-1]

            # Risk indicator
            with risk_placeholder.container():
                rcol1, rcol2, rcol3, rcol4 = st.columns(4)

                with rcol1:
                    score = latest.get('anomaly_score', 0)
                    if score > 0.6:
                        risk_level = "🔴 HIGH"
                    elif score > 0.5:
                        risk_level = "🟡 MEDIUM"
                    else:
                        risk_level = "🟢 NORMAL"
                    st.metric("Risk Level", risk_level)

                with rcol2:
                    st.metric("Anomaly Score", f"{score:.3f}")

                with rcol3:
                    st.metric("Oil Temp", f"{latest['Oil_temperature']:.1f}°C")

                with rcol4:
                    st.metric("Records Processed", f"{st.session_state.sim_idx - start_idx:,}")

            # Chart
            with chart_placeholder.container():
                hist_df = pd.DataFrame(st.session_state.history)

                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   subplot_titles=['Oil Temperature', 'Anomaly Score'],
                                   vertical_spacing=0.1)

                # Oil temperature
                fig.add_trace(
                    go.Scatter(x=hist_df['timestamp'], y=hist_df['Oil_temperature'],
                              name='Oil Temp', line=dict(color='orange')),
                    row=1, col=1
                )
                fig.add_hline(y=OIL_TEMP_THRESHOLD, line_dash="dash",
                             line_color="red", row=1, col=1)

                # Anomaly score
                fig.add_trace(
                    go.Scatter(x=hist_df['timestamp'], y=hist_df['anomaly_score'],
                              name='Anomaly Score', line=dict(color='purple')),
                    row=2, col=1
                )
                fig.add_hline(y=0.5, line_dash="dash", line_color="red", row=2, col=1)

                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            # Alerts
            with alerts_placeholder.container():
                if st.session_state.alerts:
                    st.subheader("Recent Alerts")
                    for alert in st.session_state.alerts[-5:][::-1]:
                        st.warning(alert)

        # Continue simulation
        time.sleep(0.1)
        st.rerun()

    elif not st.session_state.sim_running and st.session_state.history:
        # Show final state when paused
        latest = st.session_state.history[-1]

        with risk_placeholder.container():
            rcol1, rcol2, rcol3, rcol4 = st.columns(4)

            with rcol1:
                score = latest.get('anomaly_score', 0)
                if score > 0.6:
                    risk_level = "🔴 HIGH"
                elif score > 0.5:
                    risk_level = "🟡 MEDIUM"
                else:
                    risk_level = "🟢 NORMAL"
                st.metric("Risk Level", risk_level)

            with rcol2:
                st.metric("Anomaly Score", f"{score:.3f}")

            with rcol3:
                st.metric("Oil Temp", f"{latest['Oil_temperature']:.1f}°C")

            with rcol4:
                st.metric("Records Processed", f"{st.session_state.sim_idx - start_idx:,}")

        with chart_placeholder.container():
            hist_df = pd.DataFrame(st.session_state.history)

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               subplot_titles=['Oil Temperature', 'Anomaly Score'],
                               vertical_spacing=0.1)

            fig.add_trace(
                go.Scatter(x=hist_df['timestamp'], y=hist_df['Oil_temperature'],
                          name='Oil Temp', line=dict(color='orange')),
                row=1, col=1
            )
            fig.add_hline(y=OIL_TEMP_THRESHOLD, line_dash="dash",
                         line_color="red", row=1, col=1)

            fig.add_trace(
                go.Scatter(x=hist_df['timestamp'], y=hist_df['anomaly_score'],
                          name='Anomaly Score', line=dict(color='purple')),
                row=2, col=1
            )
            fig.add_hline(y=0.5, line_dash="dash", line_color="red", row=2, col=1)

            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with alerts_placeholder.container():
            if st.session_state.alerts:
                st.subheader("Recent Alerts")
                for alert in st.session_state.alerts[-5:][::-1]:
                    st.warning(alert)


if __name__ == "__main__":
    main()
