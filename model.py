import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pandas.errors import EmptyDataError
import joblib

import smtplib
from email.mime.text import MIMEText

def send_email(subject, message, to_emails):
    sender_email = ""
    sender_password = "" 

    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To']="".join(to_emails)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, to_emails, msg.as_string())
        return True, "Email sent successfully."
    except smtplib.SMTPException as e:
        return False, f"Error sending email: {str(e)}"

recipients = [
    "ashwinvqcon@gmail.com",
    "ss7942911@gmail.com",
    "anand19122004@gmail.com",
    "anshika.tiwari1829@gmail.com",
]



send_email(
    subject="Rockfall Alert",
    message="Warning! Rockfall detected in Zone Overburden Dump. Please take immediate action.",
    to_emails=recipients
)

def process_uploaded_file(file):

    df = pd.read_csv(file)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_')
        .str.replace(r'[^\w\s]', '', regex=True)
    )

    rename_map = {
        'factor_of_safety_fs': 'fs',
        'reinforcement_numeric': 'reinforcement',
    }

    for orig_col in rename_map:
        if orig_col in df.columns:
            df.rename(columns={orig_col: rename_map[orig_col]}, inplace=True)

    required_columns = {'fs', 'reinforcement'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    df = df.dropna(subset=['fs', 'reinforcement'])

    if 'timestamp' not in df.columns:
        df['timestamp'] = pd.date_range(datetime.now(), periods=len(df), freq='T')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['rockfall_risk'] = ((df['fs'] < 1.4) | (df['reinforcement'] < 0.4)).astype(int)

    return df


def generate_mock_data(n=100):
    np.random.seed(42)
    data = pd.DataFrame({
        'unit_weight_knm3': np.random.uniform(15, 25, n),
        'cohesion_kpa': np.random.uniform(10, 40, n),
        'internal_friction_angle': np.random.uniform(20, 45, n),
        'slope_angle': np.random.uniform(25, 60, n),
        'slope_height_m': np.random.uniform(5, 30, n),
        'pore_water_pressure_ratio': np.random.uniform(0, 1, n),
        'reinforcement_type': np.random.choice(['Drainage', 'Geosynthetics', 'Retaining Wall', 'Soil Nailing'], n),
        'reinforcement': np.random.uniform(0, 5, n),
        'fs': np.random.uniform(1, 3, n),
        'timestamp': pd.date_range(datetime.now() - timedelta(minutes=n), periods=n, freq='T'),
    })
    data['rockfall_risk'] = ((data['fs'] < 1.4) | (data['reinforcement'] < 0.4)).astype(int)
    return data


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

uploaded_file = st.sidebar.file_uploader("Upload slope stability CSV", type="csv")
data = None

if uploaded_file:
    try:
        data = process_uploaded_file(uploaded_file)
        st.sidebar.success("File uploaded and processed!")
        data.to_csv("slope_stability_dataset.csv", index=False)
    except Exception as e:
        st.sidebar.error(f"Error processing file: {e}")
        data = generate_mock_data()
        st.sidebar.warning("Using mock data instead.")
else:
    try:
        data = pd.read_csv("slope_stability_dataset.csv")
        if 'timestamp' not in data.columns:
            data['timestamp'] = pd.date_range(datetime.now(), periods=len(data), freq='T')
        else:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        st.sidebar.info("Loaded existing processed data.")
    except (FileNotFoundError, EmptyDataError, ValueError):
        data = generate_mock_data()
        st.sidebar.warning("No valid uploaded data found. Using mock data.")

st.markdown("""
<style>
section[data-testid="stSidebar"] label[data-baseweb="radio"] {
    background-color: #f0f8ff; color: #003366;
    padding: 10px 16px; border-radius: 8px;
    border: 1px solid #cce0ff; margin-bottom: 6px;
    font-size: 18px; font-weight: 600;
}
section[data-testid="stSidebar"] label[data-baseweb="radio"]:hover {
    background-color: #cce6ff; border-color: #3399ff;
}
section[data-testid="stSidebar"] label[data-baseweb="radio"][aria-checked="true"] {
    background-color: #007bff; color: white !important; border-color: #0056b3;
}
</style>
""", unsafe_allow_html=True)

section = st.sidebar.radio("Navigate to:", [
    "Dashboard", "Prediction", "Alerts", "Upload Data", "Results", "AI Assistant"
])


if section == "Dashboard":
    st.title("Slope Stability / Rockfall Dashboard")
    latest = data.iloc[-1]
    current_risk = "HIGH RISK" if latest['rockfall_risk'] == 1 else "Safe"
    active_alerts = int(data['rockfall_risk'].sum())
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

    st.subheader("Factor of Safety & Reinforcement Trends")
    col5, col6 = st.columns(2)
    with col5:
        st.line_chart(data.set_index('timestamp')['fs'])
    with col6:
        st.line_chart(data.set_index('timestamp')['reinforcement'])
    st.subheader("Site Image - Mining Area")
    st.image("mine.png", caption="Current Site Condition", use_container_width=True)



elif section == "Prediction":
    st.title("Rockfall Prediction")

    latest = data.iloc[-1]

    import os
    import joblib

    model_path = "rockfall_model_fs_reinforcement.pkl"

    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            prediction = model.predict([[latest['fs'], latest['reinforcement']]])[0]
            model_used = True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            prediction = 1 if ((latest['fs'] < 1.4) or (latest['reinforcement'] < 0.4)) else 0
            model_used = False
    else:
        st.warning("Model file not found. Using rule-based prediction.")
        prediction = 1 if ((latest['fs'] < 1.4) or (latest['reinforcement'] < 0.4)) else 0
        model_used = False


    prediction_label = "HIGH RISK" if prediction == 1 else "Safe"
    prediction_color = "#dc3545" if prediction == 1 else "#28a745"

    st.markdown(f"""
    <div style="padding:20px; border-radius:10px; background-color:#f9f9f9; border-left:10px solid {prediction_color};">
        <h2 style="color:{prediction_color}; margin-bottom:5px;">Prediction Status: {prediction_label}</h2>
        <p style="color:#666;">Based on latest slope parameters</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Factor of Safety", f"{latest['fs']:.2f}")
    with col2:
        st.metric("Reinforcement", f"{latest['reinforcement']:.2f}")
    with col3:
        st.metric("Model Used", "Yes" if model_used else "No")

    with st.expander("View Detailed Input Data"):
        st.json({
            "Timestamp": str(latest['timestamp']),
            "Factor of Safety": round(latest['fs'], 3),
            "Reinforcement": round(latest['reinforcement'], 3),
        })

    st.markdown("### Rockfall Risk Trend")

    import plotly.express as px

    risk_chart_data = data[['timestamp', 'rockfall_risk']].copy()
    risk_chart_data['Risk Label'] = risk_chart_data['rockfall_risk'].map({1: 'High Risk', 0: 'Safe'})

    fig = px.area(
        risk_chart_data,
        x='timestamp',
        y='rockfall_risk',
        color='Risk Label',
        title='Rockfall Risk Over Time',
        color_discrete_map={'High Risk': '#dc3545', 'Safe': '#28a745'},
        labels={'rockfall_risk': 'Risk Level', 'timestamp': 'Time'}
    )
    fig.update_layout(
        showlegend=True,
        height=400,
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="#ffffff"
    )
    st.plotly_chart(fig, use_container_width=True)




elif section == "Alerts":
    st.title("Real-Time Alerts Dashboard")
    st.markdown("Get insights into high-risk slope stability events with real-time filtering and visualization.")


    col1, col2 = st.columns(2)
    with col1:
        notification_method = st.selectbox("Notification Method", ["Email", "SMS", "App Notification", "None"])
        fs_threshold = st.slider("FS Threshold (lower bound)", float(data['fs'].min()), float(data['fs'].max()), value=1.4, step=0.1)
    with col2:
        date_filter = st.date_input("From Date", value=data['timestamp'].min().date())
        type_filter = st.multiselect(" Alert Type", ["Critical", "FS Alert", "Reinforcement Alert", "Info"], default=["Critical", "FS Alert", "Reinforcement Alert"])
        severity_filter = st.multiselect("Severity Level", ["High", "Moderate", "Low"], default=["High", "Moderate"])


    alerts = data[data['rockfall_risk'] == 1].copy()
    alerts['DateTime'] = pd.to_datetime(alerts['timestamp'])

    def get_type(row):
        if row['fs'] < fs_threshold and row['reinforcement'] < 0.4:
            return "Critical"
        elif row['fs'] < fs_threshold:
            return "FS Alert"
        elif row['reinforcement'] < 0.4:
            return "Reinforcement Alert"
        return "Info"

    def get_severity(fs_val):
        if fs_val < 1.0:
            return "High"
        elif fs_val < 1.4:
            return "Moderate"
        return "Low"

    alerts['Type'] = alerts.apply(get_type, axis=1)
    alerts['Severity'] = alerts['fs'].apply(get_severity)
    alerts = alerts[alerts['DateTime'].dt.date >= date_filter]

    if type_filter:
        alerts = alerts[alerts['Type'].isin(type_filter)]
    if severity_filter:
        alerts = alerts[alerts['Severity'].isin(severity_filter)]


    st.subheader("Alert Timeline")
    if not alerts.empty:
        alert_chart = alerts.copy()
        alert_chart['Alert Count'] = 1
        alert_chart = alert_chart.set_index('DateTime').resample('H')['Alert Count'].sum()
        st.line_chart(alert_chart)

    else:
        st.info("No alerts to visualize for selected filters.")

    st.subheader("Detailed Alert Log")
    if alerts.empty:
        st.success("No high-risk events detected with current filters.")
    else:
        def color_severity(sev):
            if sev == "High":
                return f"<span style='color:red;font-weight:bold;'>{sev}</span>"
            elif sev == "Moderate":
                return f"<span style='color:orange;font-weight:bold;'>{sev}</span>"
            return f"<span style='color:green;font-weight:bold;'>{sev}</span>"

        alerts_display = alerts[['DateTime', 'Type', 'Severity', 'fs', 'reinforcement']].copy()
        alerts_display['Severity'] = alerts_display['Severity'].apply(lambda x: color_severity(x))
        alerts_display = alerts_display.rename(columns={
            'DateTime': 'Timestamp',
            'Type': 'Type',
            'Severity': 'Severity',
            'fs': ' FS',
            'reinforcement': ' Reinforcement'
        })

        st.markdown(alerts_display.to_html(escape=False, index=False), unsafe_allow_html=True)

        st.subheader("Top 3 Critical Alerts")
        top_alerts = alerts.sort_values(by=['fs', 'reinforcement']).head(3)
        for i, row in top_alerts.iterrows():
            st.markdown(f"""
            <div style='background-color:#fff3cd; padding:15px; border-radius:10px; margin-bottom:10px; border-left:5px solid #ffa500'>
                <strong> {row['DateTime'].strftime('%Y-%m-%d %H:%M:%S')}</strong><br>
                <strong>Type:</strong> {row['Type']}<br>
                <strong>Severity:</strong> {row['Severity']}<br>
                <strong>FS:</strong> {row['fs']:.3f} &nbsp;|&nbsp; <strong>Reinforcement:</strong> {row['reinforcement']:.3f}
            </div>
            """, unsafe_allow_html=True)


    if st.button("Send Test Notification via Email"):
        success, msg = send_email(
            subject="Rockfall Alert",
            message="Warning! Rockfall detected in Zone Overburden Dump. Please take immediate action.",
            to_emails=recipients
    )
        if success:
            st.success(msg)
        else:
            st.error(msg)

elif section == "Upload Data":
    st.title("Upload New Slope Stability Data")
    upload_file = st.file_uploader("Choose a CSV file", type="csv", key="upload2")
    if upload_file is not None:
        try:
            new_data = process_uploaded_file(upload_file)
            new_data.to_csv("slope_stability_dataset.csv", index=False)
            st.success("File uploaded and saved successfully!")
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")
    st.subheader("Site Image - Mining Area")
    st.image("mine.png", caption="Current Site Condition", use_container_width=True)

elif section == "Results":
    st.title("Prediction Results Summary")


    total_records = len(data)
    total_alerts = int(data['rockfall_risk'].sum())
    avg_fs = data['fs'].mean()
    avg_reinf = data['reinforcement'].mean()
    risk_percent = (data['rockfall_risk'].mean()) * 100

    if risk_percent > 50:
        overall_risk = "High"
        risk_color = "error"
    elif risk_percent > 20:
        overall_risk = "Moderate"
        risk_color = "warning"
    else:
        overall_risk = "Low"
        risk_color = "success"

    analysis_date = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card(" Total Records", f"{total_records}", "success")
    with col2:
        metric_card("Total Alerts", f"{total_alerts}", "error" if total_alerts > 0 else "success")
    with col3:
        metric_card("Avg FS", f"{avg_fs:.3f}", "success" if avg_fs > 1.4 else "warning")

    col4, col5, col6 = st.columns(3)
    with col4:
        metric_card(" Avg Reinforcement", f"{avg_reinf:.3f}", "success" if avg_reinf > 0.4 else "warning")
    with col5:
        metric_card("Overall Risk Level", overall_risk, risk_color)
    with col6:
        metric_card("Last Analyzed", analysis_date, "success")

    st.markdown("### Rockfall Risk Proportion")
    st.progress(min(int(risk_percent), 100), text=f"{risk_percent:.2f}% of records are risky")

    
    st.markdown("### Risk Classification Breakdown")
    risk_counts = data['rockfall_risk'].value_counts().rename({0: "Safe", 1: "High Risk"})
    st.pyplot(
        risk_counts.plot.pie(autopct='%1.1f%%', startangle=90, shadow=True, colors=["#90ee90", "#ff4d4d"],
                             ylabel="").figure
    )


elif section == "AI Assistant":
    st.title("AI Assistant")

    prompt = st.text_area("Ask about rockfall risk, slope stability, or data insights:")
    if st.button("Get Response"):

        question = prompt.lower()
        if "risk" in question:
            risk_percent = 100 * data['rockfall_risk'].mean()
            st.write(
                f"Approximately **{risk_percent:.1f}%** of the monitored slopes currently show high rockfall risk.")
        elif "factor of safety" in question:
            avg_fs = data['fs'].mean()
            st.write(f"The average Factor of Safety in the dataset is **{avg_fs:.2f}**.")
        elif "reinforcement" in question:
            avg_reinf = data['reinforcement'].mean()
            st.write(f"The average reinforcement numeric value is **{avg_reinf:.2f}**.")
        elif "rockfall" in question and "reduce" in question:
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
            st.write(
                "Sorry, I can answer questions about risk, factor of safety, and reinforcement. Try asking about those!")


