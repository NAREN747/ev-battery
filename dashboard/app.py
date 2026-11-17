"""
Streamlit Dashboard for EV Battery Health Monitoring
Interactive dashboard for inspecting test batteries, viewing predictions, 
and getting maintenance recommendations.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import sys
from typing import Dict

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from maintenance_decision import MaintenanceDecisionEngine, create_maintenance_context, format_recommendation, DegradationSeverity, FailureRisk, MaintenanceAction

# Page config
st.set_page_config(
    page_title="EV Battery Health Monitoring",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
ROOT_DIR = Path(__file__).parent.parent
MODELS_DIR = ROOT_DIR / "models"
FEATURES_DIR = ROOT_DIR / "data" / "features"
PLOTS_DIR = ROOT_DIR / "plots"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .safety-warning {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .recommendation-high {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation-medium {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation-low {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load all necessary data."""
    enriched = pd.read_csv(FEATURES_DIR / "enriched_data.csv")
    features = pd.read_csv(FEATURES_DIR / "features.csv")
    targets = pd.read_csv(FEATURES_DIR / "targets.csv")
    
    # Load split info
    splits = joblib.load(MODELS_DIR / "results_summary.pkl")
    test_idx = splits['splits']['test']
    
    test_data = enriched.iloc[test_idx].copy()
    
    return enriched, features, targets, test_data, test_idx


@st.cache_resource
def load_models():
    """Load trained models and preprocessor."""
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
    
    models = {}
    for target in ['soh', 'rul', 'degradation_severity_encoded', 'failure_risk_encoded']:
        model_path = MODELS_DIR / f"best_{target}.pkl"
        if model_path.exists():
            models[target] = joblib.load(model_path)
    
    # Load feature names
    feature_names = []
    feature_names_path = FEATURES_DIR / "feature_names.txt"
    if feature_names_path.exists():
        with open(feature_names_path, 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
    
    return preprocessor, models, feature_names


def predict_for_battery(battery_data: pd.DataFrame, preprocessor, models, feature_names: list):
    """Make predictions for a battery's cycles using exact training feature names."""
    # Use exact training feature names and order
    X = battery_data.reindex(columns=feature_names, fill_value=0)
    
    X_processed = preprocessor.transform(X)
    
    predictions = {}
    for target, model in models.items():
        predictions[target] = model.predict(X_processed)
    
    return predictions


def create_degradation_plot(battery_data: pd.DataFrame, predictions: Dict = None):
    """Create interactive degradation curve plot."""
    battery_data = battery_data.sort_values('cycle')
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('State of Health (SOH)', 'Remaining Useful Life (RUL)', 
                       'Capacity', 'Temperature'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # SOH
    fig.add_trace(
        go.Scatter(x=battery_data['cycle'], y=battery_data['soh'], 
                  mode='lines+markers', name='Actual SOH', line=dict(color='blue', width=2)),
        row=1, col=1
    )
    if predictions and 'soh' in predictions:
        fig.add_trace(
            go.Scatter(x=battery_data['cycle'], y=predictions['soh'],
                      mode='lines+markers', name='Predicted SOH', line=dict(color='red', width=2, dash='dash')),
            row=1, col=1
        )
    fig.add_hline(y=70, line_dash="dot", line_color="red", annotation_text="EOL (70%)", row=1, col=1)
    fig.update_yaxes(title_text="SOH (%)", row=1, col=1)
    
    # RUL
    fig.add_trace(
        go.Scatter(x=battery_data['cycle'], y=battery_data['rul'],
                  mode='lines+markers', name='Actual RUL', line=dict(color='green', width=2)),
        row=1, col=2
    )
    if predictions and 'rul' in predictions:
        fig.add_trace(
            go.Scatter(x=battery_data['cycle'], y=predictions['rul'],
                      mode='lines+markers', name='Predicted RUL', line=dict(color='orange', width=2, dash='dash')),
            row=1, col=2
        )
    fig.update_yaxes(title_text="RUL (cycles)", row=1, col=2)
    
    # Capacity
    fig.add_trace(
        go.Scatter(x=battery_data['cycle'], y=battery_data['capacity'],
                  mode='lines+markers', name='Capacity', line=dict(color='purple', width=2)),
        row=2, col=1
    )
    fig.update_yaxes(title_text="Capacity (Ah)", row=2, col=1)
    fig.update_xaxes(title_text="Cycle", row=2, col=1)
    
    # Temperature
    fig.add_trace(
        go.Scatter(x=battery_data['cycle'], y=battery_data['temp_mean'],
                  mode='lines+markers', name='Mean Temp', line=dict(color='red', width=2)),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=battery_data['cycle'], y=battery_data['temp_max'],
                  mode='lines', name='Max Temp', line=dict(color='orange', width=1, dash='dot')),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=battery_data['cycle'], y=battery_data['ambient_temp'],
                  mode='lines', name='Ambient Temp', line=dict(color='gray', width=1, dash='dot')),
        row=2, col=2
    )
    fig.update_yaxes(title_text="Temperature (°C)", row=2, col=2)
    fig.update_xaxes(title_text="Cycle", row=2, col=2)
    
    fig.update_layout(height=700, showlegend=True, title_text=f"Battery {battery_data['battery_id'].iloc[0]} - Degradation Analysis")
    
    return fig


def create_actual_vs_predicted_plot(battery_data: pd.DataFrame, predictions: Dict, target: str):
    """Create actual vs predicted scatter plot."""
    actual_col = target.replace('_encoded', '')
    if actual_col not in battery_data.columns:
        actual_col = target
    
    actual = battery_data[actual_col].values
    pred = predictions.get(target, np.array([]))
    
    if len(pred) == 0:
        return None
    
    fig = go.Figure()
    
    # Scatter
    fig.add_trace(go.Scatter(
        x=actual, y=pred,
        mode='markers',
        marker=dict(size=8, color='blue', opacity=0.6),
        name='Predictions'
    ))
    
    # Perfect prediction line
    min_val = min(actual.min(), pred.min())
    max_val = max(actual.max(), pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode='lines', line=dict(color='red', dash='dash', width=2),
        name='Perfect Prediction'
    ))
    
    fig.update_layout(
        title=f'{target} - Actual vs Predicted',
        xaxis_title='Actual',
        yaxis_title='Predicted',
        height=400
    )
    
    return fig


def display_recommendations(recommendations):
    """Display maintenance recommendations with appropriate styling."""
    for i, rec in enumerate(recommendations):
        priority_class = {
            'Critical': 'safety-warning',
            'High': 'recommendation-high',
            'Medium': 'recommendation-medium',
            'Low': 'recommendation-low'
        }.get(rec.priority, 'recommendation-low')
        
        safety_icon = "🚨 " if rec.safety_concern else ""
        
        st.markdown(f"""
        <div class="{priority_class}">
            <h4>{safety_icon}{rec.action.value} (Priority: {rec.priority})</h4>
            <p><strong>Description:</strong> {rec.description}</p>
            <p><strong>Timeline:</strong> {rec.timeline}</p>
            <p><strong>Rationale:</strong></p>
            <ul>
        """, unsafe_allow_html=True)
        
        for rationale in rec.rationale:
            st.markdown(f"            <li>{rationale}</li>", unsafe_allow_html=True)
        
        st.markdown("""
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if rec.safety_concern:
            st.error("⚠️ **SAFETY HAZARD DETECTED** - Immediate action required!")


def main():
    # Header
    st.markdown('<div class="main-header">🔋 EV Battery Health Monitoring Dashboard</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data and models
    with st.spinner("Loading data and models..."):
        enriched, features, targets, test_data, test_idx = load_data()
        preprocessor, models, feature_names = load_models()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select View", [
        "📊 Battery Overview",
        "🔍 Individual Battery Analysis", 
        "📈 Model Performance",
        "⚙️ Maintenance Decisions",
        "📋 Data Explorer"
    ])
    
    # Battery selector (test batteries only)
    test_batteries = sorted(test_data['battery_id'].unique())
    selected_battery = st.sidebar.selectbox("Select Battery", test_batteries)
    
    battery_data = test_data[test_data['battery_id'] == selected_battery].sort_values('cycle')
    
    # Make predictions
    predictions = predict_for_battery(battery_data, preprocessor, models, feature_names)
    
    # Decode categorical predictions
    severity_map = {0: 'Normal', 1: 'Accelerated', 2: 'Critical'}
    risk_map = {0: 'Low', 1: 'Medium', 2: 'High'}
    
    battery_data = battery_data.copy()
    battery_data['soh_pred'] = predictions.get('soh', np.nan)
    battery_data['rul_pred'] = predictions.get('rul', np.nan)
    battery_data['severity_pred'] = [severity_map.get(int(p), 'Unknown') for p in predictions.get('degradation_severity_encoded', [])]
    battery_data['risk_pred'] = [risk_map.get(int(p), 'Unknown') for p in predictions.get('failure_risk_encoded', [])]
    
    # Current status (latest cycle)
    latest = battery_data.iloc[-1]
    
    if page == "📊 Battery Overview":
        st.header(f"Battery {selected_battery} - Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            soh_val = latest['soh']
            soh_delta = latest['soh_pred'] - latest['soh'] if 'soh_pred' in latest and not np.isnan(latest['soh_pred']) else 0
            st.metric("State of Health", f"{soh_val:.1f}%", f"{soh_delta:+.1f}% pred")
        
        with col2:
            rul_val = latest['rul']
            rul_delta = latest['rul_pred'] - latest['rul'] if 'rul_pred' in latest and not np.isnan(latest['rul_pred']) else 0
            st.metric("Remaining Useful Life", f"{rul_val:.0f} cycles", f"{rul_delta:+.0f} pred")
        
        with col3:
            severity = latest['degradation_severity']
            sev_pred = latest.get('severity_pred', 'N/A')
            st.metric("Degradation Severity", severity, f"Pred: {sev_pred}")
        
        with col4:
            risk = latest['failure_risk']
            risk_pred = latest.get('risk_pred', 'N/A')
            st.metric("Failure Risk", risk, f"Pred: {risk_pred}")
        
        # Degradation curves
        st.subheader("Degradation Curves")
        fig = create_degradation_plot(battery_data, predictions)
        st.plotly_chart(fig, use_container_width=True)
        
        # Actual vs Predicted
        st.subheader("Actual vs Predicted")
        col1, col2 = st.columns(2)
        
        with col1:
            fig_soh = create_actual_vs_predicted_plot(battery_data, predictions, 'soh')
            if fig_soh:
                st.plotly_chart(fig_soh, use_container_width=True)
        
        with col2:
            fig_rul = create_actual_vs_predicted_plot(battery_data, predictions, 'rul')
            if fig_rul:
                st.plotly_chart(fig_rul, use_container_width=True)
    
    elif page == "🔍 Individual Battery Analysis":
        st.header(f"Battery {selected_battery} - Detailed Analysis")
        
        # Cycle selector
        cycle = st.slider("Select Cycle", int(battery_data['cycle'].min()), int(battery_data['cycle'].max()), int(battery_data['cycle'].max()))
        cycle_data = battery_data[battery_data['cycle'] == cycle].iloc[0]
        
        # Current cycle metrics
        st.subheader(f"Cycle {cycle} Metrics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("SOH", f"{cycle_data['soh']:.1f}%", f"Pred: {cycle_data.get('soh_pred', 0):.1f}%")
            st.metric("Capacity", f"{cycle_data['capacity']:.4f} Ah")
            st.metric("RUL", f"{cycle_data['rul']:.0f}", f"Pred: {cycle_data.get('rul_pred', 0):.0f}")
        
        with col2:
            st.metric("Degradation Severity", cycle_data['degradation_severity'], f"Pred: {cycle_data.get('severity_pred', 'N/A')}")
            st.metric("Failure Risk", cycle_data['failure_risk'], f"Pred: {cycle_data.get('risk_pred', 'N/A')}")
            st.metric("Cycle Degradation", f"{cycle_data['cycle_degradation_pct']:.4f}%")
        
        with col3:
            st.metric("Temp (Mean/Max)", f"{cycle_data['temp_mean']:.1f}°C / {cycle_data['temp_max']:.1f}°C")
            st.metric("Voltage (Mean)", f"{cycle_data['voltage_mean']:.3f} V")
            st.metric("Current (Mean)", f"{cycle_data['current_mean']:.3f} A")
        
        # Sensor readings table
        st.subheader("Sensor Readings")
        sensor_cols = ['voltage_mean', 'voltage_min', 'voltage_max', 
                       'current_mean', 'current_min', 'current_max',
                       'temp_mean', 'temp_min', 'temp_max', 'ambient_temp',
                       'duration', 'n_measurements']
        sensor_df = cycle_data[sensor_cols].to_frame().T
        st.dataframe(sensor_df, use_container_width=True)
        
        # Historical trends
        st.subheader("Historical Trends")
        metric = st.selectbox("Select Metric", ['soh', 'rul', 'capacity', 'temp_mean', 'voltage_mean', 'current_mean'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=battery_data['cycle'], y=battery_data[metric], 
                                mode='lines+markers', name='Actual', line=dict(color='blue')))
        
        pred_col = f'{metric}_pred'
        if pred_col in battery_data.columns:
            fig.add_trace(go.Scatter(x=battery_data['cycle'], y=battery_data[pred_col],
                                    mode='lines+markers', name='Predicted', line=dict(color='red', dash='dash')))
        
        fig.update_layout(title=f"{metric.upper()} Over Cycles", xaxis_title="Cycle", yaxis_title=metric.upper())
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "⚙️ Maintenance Decisions":
        st.header(f"Battery {selected_battery} - Maintenance Recommendations")
        
        # Decision engine
        engine = MaintenanceDecisionEngine()
        
        # Context from latest cycle
        context = create_maintenance_context(
            soh=latest['soh'],
            rul=latest['rul'],
            degradation_severity=latest['degradation_severity'],
            failure_risk=latest['failure_risk'],
            temp_max=latest['temp_max'],
            temp_mean=latest['temp_mean'],
            voltage_min=latest['voltage_min'],
            current_max=latest['current_max'],
            eol_observed=latest.get('eol_observed', False),
            cycle=int(latest['cycle'])
        )
        
        # Get recommendations
        recommendations = engine.evaluate(context)
        
        # Display primary recommendation
        st.subheader("Primary Recommendation")
        primary = recommendations[0]
        
        priority_color = {
            'Critical': '🔴',
            'High': '🟠',
            'Medium': '🟡',
            'Low': '🟢'
        }.get(primary.priority, '⚪')
        
        st.markdown(f"### {priority_color} **{primary.action.value}** (Priority: {primary.priority})")
        st.write(primary.description)
        st.write(f"**Timeline:** {primary.timeline}")
        
        if primary.safety_concern:
            st.error("⚠️ **SAFETY HAZARD DETECTED** - Immediate action required!")
        
        with st.expander("Rationale"):
            for r in primary.rationale:
                st.write(f"• {r}")
        
        # All applicable recommendations
        st.subheader("All Applicable Rules")
        for i, rec in enumerate(recommendations):
            with st.expander(f"{i+1}. {rec.action.value} (Priority: {rec.priority})"):
                st.write(f"**Description:** {rec.description}")
                st.write(f"**Timeline:** {rec.timeline}")
                st.write("**Rationale:**")
                for r in rec.rationale:
                    st.write(f"  • {r}")
                if rec.safety_concern:
                    st.error("⚠️ Safety concern!")
        
        # Maintenance action guide
        st.subheader("Action Guide")
        action_guides = {
            MaintenanceAction.NO_ACTION: "Continue routine maintenance schedule. No additional action needed.",
            MaintenanceAction.MONITOR: "Increase monitoring frequency. Track SOH, temperature, and degradation rate at each charge cycle.",
            MaintenanceAction.SCHEDULE_MAINTENANCE: "Schedule preventive maintenance within the recommended timeline. Check cooling system, connections, and run diagnostics.",
            MaintenanceAction.URGENT_MAINTENANCE: "Schedule maintenance within 1 week. Battery shows signs of accelerated degradation or elevated risk.",
            MaintenanceAction.REPLACE_BATTERY: "Plan for battery replacement. Battery has reached or is approaching EOL threshold (70% SOH).",
            MaintenanceAction.SAFETY_INSPECTION: "Immediate safety inspection required. Check for thermal issues, cooling system failure, or physical damage.",
            MaintenanceAction.IMMEDIATE_REPLACEMENT: "STOP VEHICLE OPERATION IMMEDIATELY. Battery poses immediate safety hazard. Requires immediate replacement."
        }
        
        st.info(action_guides.get(primary.action, "No specific guide available."))
    
    elif page == "📈 Model Performance":
        st.header("Model Performance Summary")
        
        # Load model report
        report_path = MODELS_DIR / "model_report.txt"
        if report_path.exists():
            with open(report_path, 'r') as f:
                report = f.read()
            st.text(report)
        
        # Show plots if available
        st.subheader("Evaluation Plots")
        plot_files = list(PLOTS_DIR.glob("*_evaluation.png"))
        for plot_file in plot_files:
            st.image(str(plot_file), caption=plot_file.stem)
        
        st.subheader("Degradation Curves")
        curves_path = PLOTS_DIR / "degradation_curves.png"
        if curves_path.exists():
            st.image(str(curves_path), caption="Test Battery Degradation Curves")
    
    elif page == "📋 Data Explorer":
        st.header("Data Explorer")
        
        # Dataset summary
        st.subheader("Dataset Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", len(enriched))
        with col2:
            st.metric("Total Batteries", enriched['battery_id'].nunique())
        with col3:
            st.metric("Features", len(features.columns))
        
        # Battery summary table
        st.subheader("Battery Summary")
        battery_summary = enriched.groupby('battery_id').agg(
            cycles=('cycle', 'max'),
            initial_soh=('soh', 'first'),
            final_soh=('soh', 'last'),
            min_soh=('soh', 'min'),
            avg_temp=('temp_mean', 'mean'),
            max_temp=('temp_max', 'max'),
            degradation_severity=('degradation_severity', 'first'),
            failure_risk=('failure_risk', 'first')
        ).reset_index()
        st.dataframe(battery_summary, use_container_width=True)
        
        # Feature importance (if available)
        st.subheader("Feature Statistics")
        st.dataframe(features.describe(), use_container_width=True)
        
        # Raw data view
        if st.checkbox("Show Raw Data"):
            st.dataframe(enriched.head(100), use_container_width=True)
        
        # Download data
        csv = enriched.to_csv(index=False)
        st.download_button(
            label="Download Full Dataset as CSV",
            data=csv,
            file_name="ev_battery_data.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()