import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest

def recommendations_page():
    st.set_page_config(page_title="OptiFit Companion", layout="wide")
    st.title("🌟 Personalized Recommendations 🌟")

    # File uploader for user input
    uploaded_file = st.file_uploader("📂 Upload your dataset (CSV file)", type=["csv"])
    
    # Fallback if no file is uploaded
    if not uploaded_file:
        st.info("ℹ️ Please upload a dataset to start the analysis.")
        return
    
    # Load the dataset
    try:
        data = pd.read_csv(uploaded_file)
        st.write("📊 Uploaded Dataset Preview:")
        st.dataframe(data.head())
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return

    # Validate required columns
    required_features = ['steps', 'heart_rate']
    missing_features = [feature for feature in required_features if feature not in data.columns]
    if missing_features:
        st.error(f"⚠️ Missing required columns: {', '.join(missing_features)}")
        return

    # Anomaly detection
    model = IsolationForest(contamination=0.1, random_state=42)
    data['anomaly'] = model.fit_predict(data[required_features])

    # Calculate anomalies
    anomalies = data[data['anomaly'] == -1]
    anomaly_percentage = (len(anomalies) / len(data)) * 100

    # Display gauge chart
    st.subheader("📈 Anomaly Analysis:")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=anomaly_percentage,
        title={'text': "Anomaly Percentage"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkred"},
            'steps': [
                {'range': [0, 25], 'color': "lightgreen"},
                {'range': [25, 50], 'color': "yellow"},
                {'range': [50, 100], 'color': "red"}
            ]
        }
    ))
    st.plotly_chart(fig)

    # Recommendations Section
    st.subheader("📋 Recommendations Based on Anomalies:")
    if anomalies.empty:
        st.success("🎉 No anomalies detected! Great job!")
    else:
        # Styled Anomalies Section
        st.markdown(
            """
            <div style="background-color:#f9f7f7;padding:15px;border-radius:10px;">
                <h4 style="color:#2b2d42;">Detected Anomalies</h4>
                <p style="color:#8d99ae;">Review the following timestamps for potential issues:</p>
            </div>
            """, unsafe_allow_html=True
        )
        st.dataframe(anomalies)

        # Recommendations
        st.markdown(
            """
            <div style="background-color:#edf2f4;padding:15px;border-radius:10px;margin-top:10px;">
                <h4 style="color:#ef233c;">Recommendations</h4>
                <ul>
                    <li style="color:#2b2d42;">Increase daily activity to maintain consistent step counts.</li>
                    <li style="color:#2b2d42;">Monitor heart rate trends and consult a professional if necessary.</li>
                    <li style="color:#2b2d42;">Ensure balanced physical and mental activity daily.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True
        )
