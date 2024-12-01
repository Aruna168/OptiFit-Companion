import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
# Mock user credentials for login
USER_CREDENTIALS = {"user": "password"}

# Session State for managing user login
def initialize_session():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = ''

initialize_session()

# Sample data for mock fitness tracker
def generate_mock_data():
    data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=30, freq='D'),
        'steps': np.random.randint(5000, 15000, 30),
        'calories': np.random.randint(1500, 3000, 30),
        'sleep': np.random.randint(6, 10, 30),
        'hydration': np.random.randint(1500, 3000, 30),
        'oxygen_levels': np.random.uniform(95, 100, 30),
        'heart_rate': np.random.randint(60, 120, 30),
        'mental_health': np.random.randint(1, 10, 30)  # Scale 1 to 10 for mental health
    })
    return data


# Sidebar navigation for the hamburger menu
def sidebar_navigation():
    st.sidebar.title("Navigation")
    option = st.sidebar.radio("Go to", ["Dashboard", "Upload Data", "Recommendations","Daily Data Visualization"])
    return option

# Main dashboard function
def main_dashboard():
    st.title("OptiFit Companion")

    # Generate mock data
    data = generate_mock_data()
    
    # Display some basic statistics
    st.write("### Fitness Tracker Overview")
    st.write(f"Total Steps: {data['steps'].sum()}")
    st.write(f"Average Steps:{data['steps'].mean():.2f}")
    st.write(f"Total Calories Burned: {data['calories'].sum()}")
    st.write(f"Average Calories Burned per day:{data['calories'].mean():.2f}")
    st.write(f"Average Sleep Hours: {data['sleep'].mean():.2f}")
    st.write(f"Average Hydration (ml): {data['hydration'].mean():.2f}")
    
    # Separate Visualizations for each Metric
    st.write("### Visualization of Fitness Metrics")
    
    # Plot Steps
    fig, ax = plt.subplots(figsize=(10, 6))
    data.set_index('date')['steps'].plot(ax=ax)
    ax.set_title('Steps Over Time')
    st.pyplot(fig)

    # Plot Calories
    fig, ax = plt.subplots(figsize=(10, 6))
    data.set_index('date')['calories'].plot(ax=ax)
    ax.set_title('Calories Over Time')
    st.pyplot(fig)

    # Plot Sleep
    fig, ax = plt.subplots(figsize=(10, 6))
    data.set_index('date')['sleep'].plot(ax=ax)
    ax.set_title('Sleep Hours Over Time')
    st.pyplot(fig)

    # Plot Hydration
    fig, ax = plt.subplots(figsize=(10, 6))
    data.set_index('date')['hydration'].plot(ax=ax)
    ax.set_title('Hydration (ml) Over Time')
    st.pyplot(fig)

    # Plot Heart Rate
    fig, ax = plt.subplots(figsize=(10, 6))
    data.set_index('date')['heart_rate'].plot(ax=ax)
    ax.set_title('Heart Rate Over Time')
    st.pyplot(fig)

    # Plot Mental Health
    fig, ax = plt.subplots(figsize=(10, 6))
    data.set_index('date')['mental_health'].plot(kind='bar', ax=ax)
    ax.set_title('Mental Health Over Time')
    st.pyplot(fig)

    # Grouped Visualization of Metrics
    st.write("### Grouped Visualization of Fitness Metrics")
    
    # Grouped Bar Chart of Steps, Calories, and Sleep
    metrics = ['steps', 'calories', 'sleep']
    data_grouped = data[metrics].set_index(data['date'])
    fig, ax = plt.subplots(figsize=(12, 7))
    data_grouped.plot(kind='bar', ax=ax)
    ax.set_title('Steps, Calories, and Sleep (Grouped)')
    st.pyplot(fig)

    # Grouped Line Chart of Hydration, Heart Rate, and Mental Health
    metrics = ['hydration', 'heart_rate', 'mental_health']
    data_grouped = data[metrics].set_index(data['date'])
    fig, ax = plt.subplots(figsize=(12, 7))
    data_grouped.plot(ax=ax)
    ax.set_title('Hydration, Heart Rate, and Mental Health (Grouped)')
    st.pyplot(fig)

    # Anomaly detection with Isolation Forest
    st.write("### Anomaly Detection")
    features = ['steps', 'calories', 'sleep', 'hydration', 'oxygen_levels', 'heart_rate', 'mental_health']
    model = IsolationForest(contamination=0.1)
    data['anomaly'] = model.fit_predict(data[features])
    anomalies = data[data['anomaly'] == -1]
    st.write(f"Anomalies detected: {len(anomalies)}")
    
    # Display anomalies
    st.write("Anomalous Data Points")
    st.dataframe(anomalies)

    # Plot Anomalies (Scatter Plot)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=data, x='date', y='steps', hue='anomaly', palette={1: 'blue', -1: 'red'}, ax=ax)
    ax.set_title('Steps Anomalies')
    st.pyplot(fig)
    
    # Show recommendation based on data
    st.write("### Personalized Recommendations")
    recommendations = generate_recommendations(data)
    st.write(recommendations)

# Function to generate recommendations based on fitness data
def generate_recommendations(data):
    recommendations = ""
    
    # Steps recommendation
    if data['steps'].mean() < 10000:
        recommendations += "👟 **Increase your daily steps!** Try aiming for at least 10,000 steps a day. Walking is great for cardiovascular health. 🚶‍♂️\n"
    else:
        recommendations += "👍 **Great job on your steps!** Keep maintaining a daily target of 10,000 steps for optimal health!\n"
    
    # Sleep recommendation
    if data['sleep'].mean() < 7:
        recommendations += "💤 **Sleep more!** Aim for at least 7 hours of quality sleep each night. Better rest leads to improved recovery and energy levels.\n"
    elif data['sleep'].mean() > 9:
        recommendations += "😴 **Don't over-sleep!** While sleep is important, too much rest can also make you feel sluggish. Aim for 7-9 hours of sleep.\n"
    else:
        recommendations += "😌 **You are getting good sleep!** Keep it up! Quality sleep is key to overall well-being.\n"
    
    # Hydration recommendation
    if data['hydration'].mean() < 2000:
        recommendations += "💧 **Stay hydrated!** Aim for at least 2 liters of water per day to keep your body functioning properly.\n"
    else:
        recommendations += "🥤 **Good hydration!** Keep up the great work with your water intake.\n"
    
    # Heart Rate recommendation
    if data['heart_rate'].mean() > 100:
        recommendations += "❤️ **Your heart rate is on the high side.** It might be good to monitor stress levels or physical exertion. Consider relaxing activities like meditation or yoga.\n"
    elif data['heart_rate'].mean() < 60:
        recommendations += "❤️ **Your heart rate is low.** If you're feeling fine, this is good. Otherwise, consider checking with a doctor to ensure everything's okay.\n"
    
    # Mental Health recommendation
    if data['mental_health'].mean() < 5:
        recommendations += "🧠 **Focus on improving mental well-being.** Try stress management techniques like deep breathing, meditation, or talking to a professional.\n"
    else:
        recommendations += "🧘‍♂️ **Good mental health!** Keep up the positive mindset and self-care practices.\n"
    
    return recommendations

# Upload Data page
def upload_data():
    st.title("Upload Your Fitness Data")

    # File uploader widget
    file = st.file_uploader("Upload CSV File", type=["csv"])

    if file is not None:
        data = pd.read_csv(file)
        st.session_state.data = data  # Store the data in session state
        st.success("Dataset uploaded successfully!")
        st.dataframe(data.head())  # Display the first few rows of the uploaded dataset

    else:
        st.warning("Please upload a CSV file.")

def generate_recommendations(data):
    recommendations = []

    # 1. Steps Recommendation
    avg_steps = data['steps'].mean()
    if avg_steps < 10000:
        recommendations.append("👟 **Increase your daily steps!** Try aiming for at least 10,000 steps a day. Walking is great for cardiovascular health. 🚶‍♂️")
        recommendations.append("  *Average steps: {:.2f} steps*. Consider going for a daily walk or using a step tracker app to keep you on track.".format(avg_steps))
    else:
        recommendations.append("👍 **Great job on your steps!** Keep maintaining a daily target of 10,000 steps for optimal health!")
        recommendations.append("  *Average steps: {:.2f} steps*. You're on track with the recommended daily steps.".format(avg_steps))

    # 2. Sleep Recommendation
    avg_sleep = data['sleep'].mean()
    if avg_sleep < 7:
        recommendations.append("💤 **Sleep more!** Aim for at least 7 hours of quality sleep each night. Better rest leads to improved recovery and energy levels.")
        recommendations.append("  *Average sleep: {:.2f} hours*. Lack of sleep can hinder recovery and mental focus.".format(avg_sleep))
    elif avg_sleep > 9:
        recommendations.append("😴 **Don't over-sleep!** While sleep is important, too much rest can also make you feel sluggish. Aim for 7-9 hours of sleep.")
        recommendations.append("  *Average sleep: {:.2f} hours*. You're getting more sleep than recommended, try reducing it to 7-9 hours.".format(avg_sleep))
    else:
        recommendations.append("😌 **You are getting good sleep!** Keep it up! Quality sleep is key to overall well-being.")
        recommendations.append("  *Average sleep: {:.2f} hours*. You are in the optimal sleep range.".format(avg_sleep))

    # 3. Hydration Recommendation
    avg_hydration = data['hydration'].mean()
    if avg_hydration < 2000:
        recommendations.append("💧 **Stay hydrated!** Aim for at least 2 liters of water per day to keep your body functioning properly.")
        recommendations.append("  *Average hydration: {:.2f} ml*. Hydration is crucial for maintaining energy levels and supporting bodily functions.".format(avg_hydration))
    else:
        recommendations.append("🥤 **Good hydration!** Keep up the great work with your water intake.")
        recommendations.append("  *Average hydration: {:.2f} ml*. You're staying well-hydrated, keep it up!".format(avg_hydration))

    # 4. Heart Rate Recommendation
    avg_heart_rate = data['heart_rate'].mean()
    if avg_heart_rate > 100:
        recommendations.append("❤️ **Your heart rate is on the high side.** It might be good to monitor stress levels or physical exertion. Consider relaxing activities like meditation or yoga.")
        recommendations.append("  *Average heart rate: {:.2f} bpm*. A high resting heart rate could indicate stress or over-exertion.".format(avg_heart_rate))
    elif avg_heart_rate < 60:
        recommendations.append("❤️ **Your heart rate is low.** If you're feeling fine, this is good. Otherwise, consider checking with a doctor to ensure everything's okay.")
        recommendations.append("  *Average heart rate: {:.2f} bpm*. If you're not feeling fatigued, this is within a normal range.".format(avg_heart_rate))
    
    # 5. Mental Health Recommendation
    avg_mental_health = data['mental_health'].mean()
    if avg_mental_health < 5:
        recommendations.append("🧠 **Focus on improving mental well-being.** Try stress management techniques like deep breathing, meditation, or talking to a professional.")
        recommendations.append("  *Average mental health score: {:.2f}*. Low mental health scores indicate you may benefit from self-care and mental wellness practices.".format(avg_mental_health))
    else:
        recommendations.append("🧘‍♂️ **Good mental health!** Keep up the positive mindset and self-care practices.")
        recommendations.append("  *Average mental health score: {:.2f}*. You're on the right track with taking care of your mental health.".format(avg_mental_health))

    return recommendations


def daily_data_visualization():
    st.title("Daily Data Visualization")
    
    # Ensure the dataset is loaded
    data_file = st.file_uploader("Upload your fitness data CSV", type=["csv"])
    if data_file is not None:
        # Load the dataset
        data = pd.read_csv("C:/Users/asus/Downloads/gretel_generated_table_2024-11-30-04-58-25.csv")
        
        # Ensure the 'date' column is in datetime format
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
        else:
            st.error("The dataset must have a 'date' column!")
            return

        # Allow the user to select a specific date
        unique_dates = data['date'].dt.date.unique()
        selected_date = st.selectbox("Select a date", unique_dates)

        # Filter data for the selected date
        daily_data = data[data['date'].dt.date == selected_date]

        # Ensure the required columns exist
        required_columns = ['time', 'heart_rate', 'steps']
        missing_columns = [col for col in required_columns if col not in daily_data.columns]
        if missing_columns:
            st.error(f"Missing columns in dataset: {', '.join(missing_columns)}")
            return
        
        gauge_chart = go.Figure(go.Indicator(
                mode="gauge+number", value=daily_data['steps'].iloc[0],
                title={'text': "Steps", 'font': {'size': 24}},
                gauge={'axis': {'range': [None, 10000]},  # Assuming goal is 10,000 steps
                       'bar': {'color': "lightgreen"},
                       'steps': [{'range': [0, 5000], 'color': "yellow"},
                                 {'range': [5000, 10000], 'color': "green"}]}
            ))
        st.plotly_chart(gauge_chart)

        # Create a heatmap-compatible DataFrame
        heatmap_data = daily_data[['time', 'heart_rate']]
        heatmap_data['ValueColumn'] = 1  # Assign a default value for heatmap

        # Pivot the data for heatmap visualization
        try:
            pivoted_data = heatmap_data.pivot(index="time", columns="heart_rate", values="ValueColumn")
        except ValueError as e:
            st.error(f"Pivoting failed: {e}")
            return
        
        # Display the pivoted heatmap
        st.write("Heatmap Data (Pivoted)")
        st.write(pivoted_data)

# Plot Steps Gauge Chart with Blue and Shiny White Design
def plot_steps_gauge(daily_data):
    goal_steps = 10000  # Steps goal
    current_steps = daily_data['steps'].iloc[0]

    # Create the Gauge Chart for steps with blue and shiny white futuristic design
    gauge_chart = go.Figure(go.Indicator(
        mode="gauge+number", 
        value=current_steps,
        title={'text': "Steps", 'font': {'size': 30, 'color': "#ffffff"}},
        domain={'x': [0, 1], 'y': [0, 1]},  # Full chart area usage
        gauge={
            'axis': {'range': [None, goal_steps], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#00bfff", 'line': {'color': 'white', 'width': 3}},  # Shiny blue color for progress
            'steps': [
                {'range': [0, 5000], 'color': "#1e90ff"},  # Light blue color for low progress
                {'range': [5000, 10000], 'color': "#00bfff"},  # Bright blue color for high progress
            ],
            'threshold': {
                'line': {'color': "white", 'width': 6},  # Glowing white threshold line
                'thickness': 0.75,
                'value': current_steps
            }
        }
    ))

    # Update layout to enhance the futuristic look
    gauge_chart.update_layout(
        font={'family': "Roboto", 'color': "white", 'size': 18},
        paper_bgcolor="#1f1f1f",  # Dark background for contrast
        plot_bgcolor="#1f1f1f",  # Dark plot background for sleek look
        margin={'t': 50, 'b': 50, 'l': 50, 'r': 50},
        template='plotly_dark'  # Dark theme for modern, sleek design
    )
    
    return gauge_chart
# Plot Heart Rate Heatmap
def plot_heart_rate_heatmap(daily_data):
    # Create a heatmap-compatible DataFrame
    heatmap_data = daily_data[['time', 'heart_rate']]
    heatmap_data['ValueColumn'] = 1  # Assign a default value for heatmap
    
    # Pivot the data for heatmap visualization
    try:
        pivoted_data = heatmap_data.pivot(index="time", columns="heart_rate", values="ValueColumn")
    except ValueError as e:
        st.error(f"Pivoting failed: {e}")
        return
    
    # Plot the heatmap using seaborn
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivoted_data, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Plot Hydration Status (Donut Chart)
def plot_hydration(daily_data):
    target_hydration = 2  # Target hydration goal in liters
    consumed_hydration = daily_data['hydration'].iloc[0]  # Water consumed in liters
    
    # Calculate remaining hydration needed
    remaining_hydration = target_hydration - consumed_hydration
    
    # Plot the donut chart
    fig = go.Figure(go.Pie(
        labels=["Hydration Consumed", "Remaining Hydration"],
        values=[consumed_hydration, remaining_hydration],
        hole=0.3,  # Makes it a donut chart
        marker=dict(colors=["#00cc66", "#ff6666"]),
        title="Hydration Status"
    ))
    fig.update_traces(textinfo="percent+label")
    return fig

# Plot Sleep Status (Donut Chart)
def plot_sleep(daily_data):
    recommended_sleep = 8  # Recommended sleep in hours
    actual_sleep = daily_data['sleep'].iloc[0]  # Actual sleep in hours
    
    # Calculate remaining sleep needed
    remaining_sleep = recommended_sleep - actual_sleep
    
    # Plot the donut chart
    fig = go.Figure(go.Pie(
        labels=["Sleep Achieved", "Remaining Sleep"],
        values=[actual_sleep, remaining_sleep],
        hole=0.3,  # Makes it a donut chart
        marker=dict(colors=["#1f77b4", "#ff6666"]),
        title="Sleep Status"
    ))
    fig.update_traces(textinfo="percent+label")
    return fig

# Main function for Daily Data Visualization
def daily_data_visualization():
    st.title("Daily Data Visualization")
    
    # Ensure the dataset is loaded
    data_file = st.file_uploader("Upload your fitness data CSV", type=["csv"])
    if data_file is not None:
        # Load the dataset
        data = pd.read_csv(data_file)
        
        # Ensure the 'date' column is in datetime format
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
        else:
            st.error("The dataset must have a 'date' column!")
            return

        # Allow the user to select a specific date
        unique_dates = data['date'].dt.date.unique()
        selected_date = st.selectbox("Select a date", unique_dates)

        # Filter data for the selected date
        daily_data = data[data['date'].dt.date == selected_date]

        # Ensure the required columns exist
        required_columns = ['time', 'heart_rate', 'steps', 'hydration', 'sleep']
        missing_columns = [col for col in required_columns if col not in daily_data.columns]
        if missing_columns:
            st.error(f"Missing columns in dataset: {', '.join(missing_columns)}")
            return

        # Display visualizations
        st.write("### Hydration Status")
        st.plotly_chart(plot_hydration(daily_data))

        st.write("### Sleep Status")
        st.plotly_chart(plot_sleep(daily_data))

        st.write("### Steps Visualization")
        st.plotly_chart(plot_steps_gauge(daily_data))

        st.write("### Heart Rate Heatmap")
        plot_heart_rate_heatmap(daily_data)

        # Display other metrics
        st.write("### Daily Metrics")
        avg_steps = daily_data['steps'].mean()
        st.metric("Average Steps", f"{avg_steps:.2f}")

        # Example of a treemap (if 'Steps' is available per hour)
        st.write("Treemap Visualization")
        if 'steps' in daily_data.columns:
            fig = px.treemap(daily_data, path=['time'], values='steps', title="Steps Distribution")
            st.plotly_chart(fig)
        else:
            st.warning("Steps data is not available for treemap.")

# Anomaly detection with Isolation Forest (revised)
def recommendations_page():
    st.title("OptiFit Companion - Personalized Recommendations")

    # Check if data is available
    if 'data' in st.session_state:
        data = st.session_state.data
        st.write("Here are your personalized recommendations based on your uploaded data:")
        recommendations = generate_recommendations(data)

        # Display recommendations as bullet points
        for rec in recommendations:
            st.write(f"• {rec}")

        # Visualize the data
        st.write("### Visualization of Your Fitness Metrics and Anomalies")

        # Display a line chart for overall fitness metrics
        st.line_chart(data[['steps', 'calories', 'sleep', 'hydration', 'heart_rate']])

        # Show anomaly detection results (if any)
        st.write("### Anomaly Detection Results")
        anomalies = data[data['anomaly'] == -1]  # Correct this line
        if not anomalies.empty:
            st.write(f"Anomalies detected: {len(anomalies)}")
            st.dataframe(anomalies)

            # Visualize anomalies in a scatter plot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=data, x='date', y='steps', hue='anomaly', palette={1: 'blue', -1: 'red'}, ax=ax)
            ax.set_title('Detected Anomalies in Steps')
            st.pyplot(fig)
        else:
            st.write("No anomalies detected in your data.")
# Sidebar navigation for the hamburger menu
option = sidebar_navigation()

if option == "Dashboard":
    main_dashboard()
elif option == "Upload Data":
    upload_data()
elif option == "Recommendations":
    recommendations_page()
elif option == "Daily Data Visualization":
    daily_data_visualization()
