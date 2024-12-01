import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest

# Sidebar Navigation
st.sidebar.title("Fitness Tracker App")
app_mode = st.sidebar.radio("Select a Section", ["Home", "Fitness Tracker Overview", "Recommendations", "Anomaly Detection", "Upload Data"])

# Function to upload file
def upload_file():
    uploaded_file = st.file_uploader("Upload your fitness tracker data (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.write(data.head())
        return data
    return None

# Mock data for testing when no file is uploaded
def get_mock_data():
    # Create mock data as a DataFrame
    data = {
        "date": pd.date_range(start="2024-01-01", periods=10, freq='D'),
        "steps": [10000, 12000, 8000, 9000, 11000, 9500, 12000, 11500, 13000, 9500],
        "calories": [2000, 2200, 1800, 1900, 2100, 1950, 2200, 2150, 2300, 2000],
        "sleep": [8, 7, 6.5, 7.5, 8.5, 7, 6.5, 7.5, 8, 7],
        "hydration": [2500, 2300, 2200, 2400, 2500, 2400, 2300, 2200, 2100, 2400],
        "heart_rate": [70, 75, 80, 72, 68, 74, 71, 70, 78, 76],
        "mental_health": [7, 8, 6, 5, 7, 6, 8, 7, 6, 7],
    }
    return pd.DataFrame(data)

# Display Fitness Tracker Overview
def display_overview(data):
    st.write("### Fitness Tracker Overview")
    st.write(f"**Total Steps**: {data['steps'].sum()}")
    st.write(f"**Average Steps**: {data['steps'].mean():.2f}")
    st.write(f"**Total Calories Burned**: {data['calories'].sum()}")
    st.write(f"**Average Calories Burned per day**: {data['calories'].mean():.2f}")
    st.write(f"**Average Sleep Hours**: {data['sleep'].mean():.2f}")
    st.write(f"**Average Hydration (ml)**: {data['hydration'].mean():.2f}")

# Generate Fitness Recommendations
def generate_recommendations(data):
    recommendations = ""

    # Steps recommendation
    if data['steps'].mean() < 10000:
        recommendations += "👟 **Increase your daily steps!** Try aiming for at least 10,000 steps a day. 🚶‍♂️\n"
    else:
        recommendations += "👍 **Great job on your steps!** Keep maintaining a daily target of 10,000 steps!\n"

    # Sleep recommendation
    if data['sleep'].mean() < 7:
        recommendations += "💤 **Sleep more!** Aim for at least 7 hours of quality sleep each night.\n"
    elif data['sleep'].mean() > 9:
        recommendations += "😴 **Don't over-sleep!** Aim for 7-9 hours of sleep.\n"
    else:
        recommendations += "😌 **You're getting good sleep!** Keep it up.\n"

    # Hydration recommendation
    if data['hydration'].mean() < 2000:
        recommendations += "💧 **Stay hydrated!** Aim for at least 2 liters of water per day.\n"
    else:
        recommendations += "🥤 **Good hydration!** Keep it up.\n"

    # Mental Health recommendation
    if data['mental_health'].mean() < 5:
        recommendations += "🧠 **Focus on improving mental well-being.** Consider stress management.\n"
    else:
        recommendations += "🧘‍♂️ **Good mental health!** Keep up the positive mindset.\n"

    return recommendations

# Anomaly Detection (Isolation Forest)
def detect_anomalies(data):
    features = data[['steps', 'calories', 'sleep', 'hydration', 'heart_rate', 'mental_health']]
    model = IsolationForest(contamination=0.1)
    data['anomaly'] = model.fit_predict(features)
    anomalies = data[data['anomaly'] == -1]
    return anomalies

# Plot data visualization
def plot_data(data):
    # Plot Steps over time
    fig_steps = px.line(data, x='date', y='steps', title='Steps Over Time')
    st.plotly_chart(fig_steps)

    # Plot Sleep over time
    fig_sleep = px.line(data, x='date', y='sleep', title='Sleep Over Time')
    st.plotly_chart(fig_sleep)

# Main app logic
if app_mode == "Home":
    st.write("# Welcome to the Fitness Tracker App")
    st.write("Please use the sidebar to navigate to different sections. Upload your data to see personalized insights!")

elif app_mode == "Fitness Tracker Overview":
    # Upload file or use mock data
    data = upload_file() or get_mock_data()
    display_overview(data)

elif app_mode == "Recommendations":
    # Upload file or use mock data
    data = upload_file() or get_mock_data()
    recommendations = generate_recommendations(data)
    st.write("### Fitness Recommendations")
    st.write(recommendations)

elif app_mode == "Anomaly Detection":
    # Upload file or use mock data
    data = upload_file() or get_mock_data()
    anomalies = detect_anomalies(data)
    st.write("### Anomalies Detected")
    st.write(anomalies)

elif app_mode == "Upload Data":
    st.write("### Please Upload Your Fitness Tracker Data (CSV)")
    data = upload_file()

