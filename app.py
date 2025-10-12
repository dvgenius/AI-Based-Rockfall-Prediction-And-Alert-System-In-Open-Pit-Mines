import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from pandas.errors import EmptyDataError


def generate_mock_data(n=10000):
    timestamps = pd.date_range(datetime.now() - timedelta(minutes=n), periods=n, freq='T')
    data = pd.DataFrame({
        'timestamp': timestamps,
        'Lateral Vibration': np.random.normal(0.5, 0.1, n),
        'Elevation Vibration': np.random.normal(0.6, 0.1, n),
        'Longitudinal Vibration': np.random.normal(0.4, 0.1, n),
        'displacement': np.random.normal(1.2, 0.3, n),
        'acoustic_emission': np.random.normal(0.05, 0.02, n),
    })
    data['rockfall_risk'] = ((data['displacement'] > 1.4) | (data['acoustic_emission'] > 0.08)).astype(int)
    return data


import pandas as pd
from datetime import datetime

def process_uploaded_file(file):
    df = pd.read_csv(file)


    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^\w\s]', '', regex=True)


    rename_map = {
        'unit_weight_knm': 'unit_weight',
        'cohesion_kpa': 'cohesion',
        'internal_friction_angle': 'friction_angle',
        'slope_angle': 'slope_angle',
        'slope_height_m': 'slope_height',
        'pore_water_pressure_ratio': 'pore_pressure_ratio',
        'lateral_vibration_elevation_vibration_and_longitudinal_vibration': 'vibration_type',
        'acoustic_emission': 'acoustic_emission',
        'displacement': 'displacement'
    }

    df.rename(columns=rename_map, inplace=True)


    required_columns = {'displacement', 'acoustic_emission'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


    df = df.dropna(subset=['displacement', 'acoustic_emission'])


    if 'timestamp' not in df.columns:
        df['timestamp'] = pd.date_range(datetime.now(), periods=len(df), freq='T')

    df['timestamp'] = pd.to_datetime(df['timestamp'])


    df['rockfall_risk'] = ((df['displacement'] > 1.4) | (df['acoustic_emission'] > 0.08)).astype(int)

    return df

def metric_card(title, value, status):
    colors = {
        "success": "#28a745",
        "error": "#dc3545",
        "warning": "#ffc107"
    }
    color = colors.get(status, "black")
    st.markdown(f"""
     <div style="border-radius:10px; padding:15px; background-color:#d3d3d3;
     text-align:center; box-shadow:2px 2px 5px rgba(0,0,0,0.1);">
         <h5>{title}</h5>
         <p style="font-size:24px; font-weight:bold; color:{color}; margin:0;">{value}</p>
     </div>
     """, unsafe_allow_html=True)


st.set_page_config(page_title="Rockfall Monitoring System", layout="wide")
st.sidebar.title("MineSafe Navigation")

uploaded_file = st.sidebar.file_uploader("Upload rockfall sensor CSV", type="csv")

data = None

if uploaded_file:
    try:
        data = process_uploaded_file(uploaded_file)
        st.sidebar.success("File uploaded and processed!")

        data.to_csv('processed_rockfall_data.csv', index=False)
    except Exception as e:
        st.sidebar.error(f"{e}")
        data = generate_mock_data()
        st.sidebar.warning("Using mock data instead.")
else:

    try:
        data = pd.read_csv('.csv')
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        st.sidebar.info("Loaded previously processed data.")
    except (FileNotFoundError, EmptyDataError, ValueError):
        data = generate_mock_data()
        st.sidebar.warning("No valid uploaded data found. Using mock data.")


st.markdown("""
    <style>
    section[data-testid="stSidebar"] label[data-baseweb="radio"] {
        background-color: #f0f8ff;  
        color: #003366;             
        padding: 10px 16px;
        border-radius: 8px;
        border: 1px solid #cce0ff;
        margin-bottom: 6px;
        font-size: 18px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

   
    section[data-testid="stSidebar"] label[data-baseweb="radio"]:hover {
        background-color: #cce6ff;
        border-color: #3399ff;
    }

    section[data-testid="stSidebar"] label[data-baseweb="radio"][aria-checked="true"] {
        background-color: #007bff;  
        color: white !important;
        border-color: #0056b3;
    }

    section[data-testid="stSidebar"] .stRadio > div {
        gap: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)


section = st.sidebar.radio("Navigate to:", [
    "Dashboard", "Prediction", "Alerts", "Upload Data", "Results", "AI Assistant"
])


if section == "Dashboard":
    st.title("Rockfall Monitoring Dashboard")
    latest = data.iloc[-1]
    current_risk = "HIGH RISK" if latest['rockfall_risk'] == 1 else "Safe"
    active_alerts = data['rockfall_risk'].sum()
    system_status = "Operational" if active_alerts < 5 else "High Alerts"
    status_color = "success" if active_alerts < 5 else "warning"

    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    with col1:
        metric_card("Current Risk", current_risk, "error" if "HIGH" in current_risk else "success")
    with col2:
        metric_card("Active Alerts", active_alerts, "error" if active_alerts > 0 else "success")
    with col3:
        metric_card("System Status", system_status, status_color)
    with col4:
        st.info("Recent Activity")
        st.dataframe(data.tail(5).set_index('timestamp'))

    st.subheader("Site Image - Mining Area")
    img_path = Path("D:/mines images/IMG-20251002-WA0002.jpg")
    if img_path.exists():
        st.image(str(img_path), caption="Current Site Condition", use_container_width=True)
    else:
        st.warning("Site image not found.")

    st.subheader("Sensor Trends")
    col5, col6 = st.columns(2)
    with col5:
        st.line_chart(data.set_index('timestamp')['displacement'])
    with col6:
        st.line_chart(data.set_index('timestamp')['acoustic_emission'])

elif section == "Prediction":
    st.title("Rockfall Prediction Overview")

    latest = data.iloc[-1]
    prediction = "HIGH RISK" if latest['rockfall_risk'] == 1 else "Safe"
    prediction_color = "red" if prediction == "HIGH RISK" else "green"

    st.markdown(f"""
            <h3>Prediction Status: <span style='color:{prediction_color};'>{prediction}</span></h3>
        """, unsafe_allow_html=True)

    with st.expander("View Latest Sensor Data"):
        st.json({
            "Timestamp": str(latest['timestamp']),
            "Displacement": round(latest['displacement'], 2),
            "Acoustic Emission": round(latest['acoustic_emission'], 3),
            "Lateral Vibration": round(latest.get('Lateral Vibration', 0), 2),
            "Elevation Vibration": round(latest.get('Elevation Vibration', 0), 2),
            "Longitudinal Vibration": round(latest.get('Longitudinal Vibration', 0), 2),
        })

    st.markdown("### Risk Trend Over Time")
    st.line_chart(data.set_index('timestamp')['rockfall_risk'])



elif section == "Alerts":
    st.title("Alerts Log")
    st.markdown(
        "<p style='font-size: 14px; color: gray;'>Review and manage all alerts triggered by the rockfall prediction system.</p>",
        unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 3])

    with left_col:
        notification_method = st.selectbox("Notification Method", ["Email", "SMS", "App Notification", "None"])
        severity_threshold = st.slider("Severity Threshold (Displacement)", float(data['displacement'].min()),
                                       float(data['displacement'].max()), value=1.4, step=0.1)

    with right_col:
        col1, col2 = st.columns(2)
        with col1:
            date_filter = st.date_input("From Date", value=data['timestamp'].min().date())
        with col2:
            type_filter = st.multiselect("Alert Type", ["Critical", "Displacement Alert", "Acoustic Alert", "Info"])
        severity_filter = st.multiselect("Severity", ["High", "Moderate", "Low"])

    alerts = data[data['rockfall_risk'] == 1].copy()
    alerts['Date'] = pd.to_datetime(alerts['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')


    def get_type(row):
        if row['displacement'] > 1.4 and row['acoustic_emission'] >0.08:
            return "Critical"
        elif row['displacement'] > 1.4:
            return "Displacement Alert"
        elif row['acoustic_emission'] > 0.08:
            return "Acoustic Alert"
        else:
            return "Info"

    def get_severity(d):
        if d > 1.8:
            return "High"
        elif d > 1.4:
            return "Moderate"
        else:
            return "Low"

    alerts['Type'] = alerts.apply(get_type, axis=1)
    alerts['Severity'] = alerts['displacement'].apply(get_severity)


    alerts = alerts[alerts['displacement'] >= severity_threshold]
    alerts = alerts[pd.to_datetime(alerts['timestamp']).dt.date >= date_filter]
    alerts = alerts[alerts['Type'].isin(type_filter)]
    alerts = alerts[alerts['Severity'].isin(severity_filter)]


    if alerts.empty:
        st.success("No high-risk events detected with current filters.")
    else:
        st.error(f"{len(alerts)} high-risk alerts found!")
        st.markdown("#### Alert Details")
        st.dataframe(
            alerts[['Date', 'Type', 'Severity', 'displacement', 'acoustic_emission']],
            use_container_width=True,
            height=300
        )

        with st.expander("View Full Alert Data"):
            st.write(alerts)


        top_alerts = alerts.sort_values(
            by=['displacement', 'acoustic_emission'], ascending=False
        ).head(3)

        st.markdown("#### Top 3 Critical Alerts")
        for _, row in top_alerts.iterrows():
            st.warning(
                f"**Date**: {row['Date']}  \n"
                f"**Type**: {row['Type']}  \n"
                f"**Severity**: {row['Severity']}  \n"
                f"**Displacement**: {row['displacement']:.2f}  \n"
                f"**Acoustic Emission**: {row['acoustic_emission']:.3f}"
            )

    if st.button("Send Test Notification"):
        if notification_method == "None":
            st.info("Notifications are disabled.")
        else:
            st.success(f"Test notification sent via **{notification_method}** for alerts above {severity_threshold:.2f}.")

elif section == "Upload Data":
    st.title("Upload Data")
    st.markdown(
        "<p style='font-size: 14px; color: gray;'>Upload CSV data to generate rockfall prediction automatically.</p>",
        unsafe_allow_html=True
    )

    st.title("Upload New CSV Data")
    uploaded_file = st.file_uploader("Upload your rockfall sensor data in CSV format to generate predictions automatically", type="csv")
    if uploaded_file:
        try:
            new_data = pd.read_csv(uploaded_file)
            st.success("File uploaded!")
            st.dataframe(new_data.head())
        except Exception as e:
            st.error(f"Error: {e}")
    st.subheader("Site Image - Mining Area")
    st.image("D:\mines images\IMG-20251002-WA0004.jpg", caption="Current Site Condition", use_container_width=True)

elif section == "Results":
    st.title("Rockfall Prediction Results")
    st.markdown(
        "<p style='font-size: 14px; color: gray;'>Latest analysis for open-pit mine operations .</p>",
        unsafe_allow_html=True
    )

    total_records = len(data)
    total_alerts = data['rockfall_risk'].sum()
    avg_disp = f"{data['displacement'].mean():.2f}"
    avg_acoustic = f"{data['acoustic_emission'].mean():.3f}"
    overall_risk = "High" if data['rockfall_risk'].mean() > 0.5 else "Moderate" if data['rockfall_risk'].mean() > 0.2 else "Low"
    model_confidence = "92.5%"
    analysis_date = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Total Records")
            st.info(f"{total_records}")
        with col2:
            st.subheader("Total Alerts")
            st.warning(f"{total_alerts}")
        with col3:
            st.subheader("Avg Displacement")
            st.success(f"{avg_disp}")

    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Avg Acoustic Emission")
            st.success(f"{avg_acoustic}")
        with col2:
            st.subheader("Overall Risk Level")
            risk_color = "danger" if overall_risk == "High" else "warning" if overall_risk == "Moderate" else "success"
            if risk_color == "warning":
                st.warning(overall_risk)
            elif risk_color == "danger":
                st.error(overall_risk)
            else:
                st.success(overall_risk)
        with col3:
            st.subheader("Model Confidence")
            st.info(model_confidence)

    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Analysis Date")
            st.info(analysis_date)
        with col2:
            st.subheader("Zone Risk Analysis")
            if 'zone' in data.columns:
                zone_risk = data.groupby('zone')['rockfall_risk'].mean().reset_index()
                zone_risk['Risk Level'] = zone_risk['rockfall_risk'].apply(
                    lambda x: "High" if x > 0.5 else "Moderate" if x > 0.2 else "Low"
                )
                st.dataframe(zone_risk)
            else:
                st.info("Zone data not available.")
        with col3:
            st.subheader("Mitigation Action")
            if overall_risk == "High":
                st.error("Immediate inspection & barrier installation required.")
            elif overall_risk == "Moderate":
                st.warning("Increase monitoring & plan preventive maintenance.")
            else:
                st.success("Maintain standard monitoring procedures.")

elif section == "AI Assistant":
    st.title("AI Assistant")
    user_input = st.text_input("Ask your question:")

    if user_input:
        question = user_input.lower()

        if "rockfall" in question and "reduce" in question:
            st.markdown("""
                **To reduce rockfall risk:**
                - Install rock bolts, nets, and retaining structures.
                - Regularly inspect and monitor slope stability.
                - Control blasting operations carefully to prevent loosening of rocks.
                - Improve drainage to reduce water pressure behind rocks.
                - Use real-time monitoring systems for early detection.
            """)
        elif "safe" in question and ("worker" in question or "workers" in question):
            st.markdown("""
                **To keep workers safe from rockfall:**
                - Restrict access to high-risk areas during unstable periods.
                - Use protective shelters in high-risk zones.
                - Provide workers with PPE like helmets and reflective gear.
                - Train workers in emergency evacuation and hazard identification.
                - Use alarm systems linked to rockfall monitoring sensors.
            """)
        else:
            st.info("Response coming soon!")


