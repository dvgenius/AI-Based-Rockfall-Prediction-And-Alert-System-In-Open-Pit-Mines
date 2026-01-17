📌 Project Overview

Rockfall and slope instability are critical safety hazards in open pit mining operations. Unexpected slope failures can lead to loss of life, equipment damage, and major production delays. Traditional monitoring methods are mostly reactive and rely on manual inspections or static thresholds.

This project presents an AI-Based Rockfall Prediction and Alert System that combines machine learning, data analytics, and real-time visualization to proactively assess slope stability and generate early warnings

🚩 Problem Statement

Open pit mines face rockfall risks due to:

Low Factor of Safety (FS)

Insufficient slope reinforcement

Environmental and geological variations

Delayed detection using manual methods

There is a need for a predictive, automated, and intelligent system that can analyze slope stability parameters and issue timely alerts.

💡 Proposed Solution

The system uses historical and real-time slope stability data to:

Train a machine learning model (in Jupyter Notebook)

Predict rockfall risk based on FS and reinforcement values

Display results on an interactive Streamlit dashboard

Generate alerts and send email notifications during high-risk conditions

The trained model is saved and later loaded into the Streamlit application for real-time inference.

🧠 Machine Learning Model (Jupyter Notebook)

Model training is performed in a Jupyter Notebook

Data preprocessing, feature selection, and labeling are handled offline

Algorithms such as Logistic Regression / Random Forest / SVM (as used) are trained

The final trained model is saved using Joblib

Saved model file:

rockfall_model_fs_reinforcement.pkl


The Streamlit app uses this model for prediction, with a rule-based fallback if the model file is unavailable

✨ Key Features

📂 CSV-based slope stability data upload

🤖 AI-based rockfall prediction (ML + rule-based backup)

📊 Interactive Streamlit dashboard

📈 FS & reinforcement trend visualization

🚨 Alert classification with severity levels

📧 Email notification system

🧪 Mock data generation for demo/testing

💬 AI Assistant for safety insights

🛠️ Technology Stack

Programming & Frameworks

Python 3.x

Streamlit

Jupyter Notebook

Data Processing & Visualization

Pandas

NumPy

Plotly

Matplotlib

Machine Learning

Scikit-learn

Joblib

Alerting

SMTP (Email Notifications)

🏗️ System Architecture

Historical slope data used to train ML model (Jupyter Notebook)

Trained model saved as .pkl

Streamlit app loads model

User uploads CSV or mock data is generated

Risk prediction is performed

Dashboard visualizes trends and alerts

Email notification sent if risk is high


⚙️ Installation & Setup
1 Install Dependencies
pip install -r requirements.txt

2 Run Streamlit App
streamlit run app.py

📄 Input CSV Requirements

Mandatory columns:

fs → Factor of Safety

reinforcement → Reinforcement numeric value

Optional:

timestamp (auto-generated if missing)

🚨 Alert Logic

High Risk

FS < 1.4 OR Reinforcement < 0.4

Alerts are classified by:

Type (FS / Reinforcement / Critical)

Severity (High / Moderate / Low)

Email alert is triggered for critical conditions

📸 Screenshots
## 📸 Application Screenshots

### 📊 Dashboard

(<img width="1920" height="1200" alt="dashboard" src="https://github.com/user-attachments/assets/9fb9569c-5dad-4ee7-8a59-e0bf35a16841" />
)

### 🤖 Prediction

(<img width="1920" height="1200" alt="Screenshot 2025-10-09 002903" src="https://github.com/user-attachments/assets/6fdd7335-cc35-4bc9-92cd-1b80c1a8e0f0" />
)

### 🚨 Alerts

(<img width="1920" height="1200" alt="Screenshot 2025-10-18 210911" src="https://github.com/user-attachments/assets/23960da9-05a6-4391-a8a8-cd25dc5e3ae5" />
)
(<img width="1920" height="1200" alt="Screenshot 2025-10-18 210933" src="https://github.com/user-attachments/assets/4f507c50-4c4e-4d4d-b5f2-d8f09778e77c" />
)
(<img width="1920" height="1200" alt="Screenshot 2025-10-18 210959" src="https://github.com/user-attachments/assets/99093a92-842d-4fc5-9362-a82b8b69db89" />
)
### 📂 Upload Data
(<img width="1920" height="1200" alt="Screenshot 2025-10-18 211013" src="https://github.com/user-attachments/assets/363652c3-14f5-4349-b540-2c90618ca306" />
)

### 📈 Results
(<img width="1920" height="1200" alt="Screenshot 2025-10-18 211026" src="https://github.com/user-attachments/assets/06a872b3-3bdc-414f-8c4c-8d8a5bfe353e" />
)

### 💬 AI Assistant
(<img width="1920" height="1200" alt="Screenshot 2025-10-18 211052" src="https://github.com/user-attachments/assets/a35d9690-8fac-4f86-8773-2a22506bde39" />
)

⚠️ Limitations

Model accuracy depends on training data quality

Email requires SMTP configuration

Rule-based fallback may cause conservative alerts

No real IoT sensor integration (currently simulated)

🚀 Future Scope

IoT sensor integration

Drone & image-based slope monitoring

Digital twin of open pit mines

Edge AI deployment

SMS and mobile app alerts

Multi-mine centralized monitoring


📜 License

This project is licensed under the MIT License.

