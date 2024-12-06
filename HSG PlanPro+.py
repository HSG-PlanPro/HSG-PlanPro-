# The whole Azure OAuth Process was done with assistance of Chat-GPT

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import statistics as stat
import joblib
import numpy as np
import matplotlib.pyplot as plt
import tempfile

st.set_page_config(
    page_title="HSG PlanPro+",
    page_icon="🗓 - Scheduler",
    layout='wide'
)

st.title('HSG PlanPro+')
st.subheader('The Smart Scheduling Assistant for HSG Students')

st.write('Many students struggle with organizing their time. We believe that visualizing personal organization and time management goes a long way in building a mental framework of how time is used.')
         
st.write('This is why we developped PlanPro+. Our software allows you to interact with your Outlook calendar and provides you with multiple visual representations of your calendar.')
         
st.write(""""What's new?" you ask. Well, we allow you to define a timeframe and visualize how your time is allocated within it and we show you your sleep schedule.""")
         
st.write('But most importantly, we study your time organization and your sleep and use machine learning models to make recommendations about **how you could optimize your life**. Hooked?')

DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
today = datetime.today()

def calculate_sleep_deficit(age, avg_sleep):
    """Calculate the sleep deficit based on age."""
    recommended = 9 if age <= 12 else 8 if age <= 18 else 7
    deficit = max(0, recommended - avg_sleep)
    return f"You should sleep {deficit:.1f} more hours per night." if deficit > 0 else "You are well-rested!"

st.title('Personal Information')
st.subheader('1. Start by giving us some personal information. We can already give you a preliminary interactive overview of your sleep.')
with st.form(key="personal_info"):
    firstname = st.text_input("First Name")
    lastname = st.text_input("Last Name")
    date_of_birth = st.date_input("Date of Birth", min_value=datetime(1980, 1, 1))
    daily_steps = st.slider('Average Daily Steps in a Full Week', 0, 30000, 5000)

    # Create columns for each day of the week
    cols = st.columns(len(DAYS))
    
    # Input fields for each day in separate columns
    sleep_hours = {day: cols[i].number_input(day, min_value=0, max_value=24) for i, day in enumerate(DAYS)}
    
    if st.form_submit_button("Analyze Sleep"):
        # Calculate statistics
        avg_sleep = stat.mean(sleep_hours.values())
        total_sleep = sum(sleep_hours.values())
        
        # Display results
        st.write(f"Welcome, {firstname}!")
        st.write(f"Average Sleep: {avg_sleep:.2f} hours")
        st.write(f"Total Weekly Sleep: {total_sleep} hours")
        st.write(calculate_sleep_deficit(datetime.now().year - 1990, avg_sleep))  # Replace 1990 with a variable for date_of_birth
        deficit_message = calculate_sleep_deficit(datetime.now().year - 1990, avg_sleep)

        # Create a DataFrame with explicit ordering
        sleep_df = pd.DataFrame({
            "Day": DAYS,
            "Sleep Hours": [sleep_hours[day] for day in DAYS]  # Ensure days are ordered
        })

        # Explicitly set categorical order
        sleep_df['Day'] = pd.Categorical(sleep_df['Day'], categories=DAYS, ordered=True)

        # Plot bar chart
        st.bar_chart(sleep_df.set_index("Day"))

        #Put my variables in a session_state to use in another page for pdf report
        st.session_state['firstname'] = firstname
        st.session_state['lastname'] = lastname
        st.session_state['avg_sleep'] = avg_sleep
        st.session_state['deficit_message'] = deficit_message

        ml_daily_steps = daily_steps
        ml_sleep_duration = stat.mean(sleep_hours.values())
        ml_age = today.year - date_of_birth.year - ((today.month, today.day) < (date_of_birth.month, date_of_birth.day))

        #Sleep Quality Model
        model = joblib.load('sleep_quality_model.pkl')


        # Prepare the input data for scaling
        input_data = np.array([[ml_age, ml_sleep_duration, ml_daily_steps]])
        prediction = model.predict(input_data)
        sleep_message = (f"Predicted Sleep Quality: {prediction[0]:.2f}")

        st.session_state['daily_steps'] = daily_steps
        st.session_state['ml_sleep_duration'] = ml_sleep_duration
        st.session_state['ml_age'] = ml_age
        st.session_state['sleep_message'] = sleep_message
        st.session_state['sleep_prediction'] = prediction[0]

        st.subheader("Great! Now you already have a basic understanding of your sleep. Go to the next page: **Calendar and PlanPro+**. That's where the *real* fun begins.")

# Create a DataFrame with the sleep hours
sleep_df = pd.DataFrame({
    "Day": DAYS,
    "Sleep Hours": [sleep_hours[day] for day in DAYS]
})

# Set the 'Day' column as categorical for correct ordering
sleep_df['Day'] = pd.Categorical(sleep_df['Day'], categories=DAYS, ordered=True)

# Plot bar chart using matplotlib
fig, ax = plt.subplots(figsize=(13, 6))  # Customize the figure size
ax.bar(sleep_df['Day'], sleep_df['Sleep Hours'], color='skyblue')

# Add titles and labels
ax.set_title("Sleep Hours Per Night", fontsize=14)
ax.set_xlabel("Day", fontsize=12)
ax.set_ylabel("Sleep Hours", fontsize=12)

# Create a temporary file to store the chart
with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
    chart_image_path = temp_file.name
    fig.savefig(chart_image_path)

st.session_state['chart_image_path'] = chart_image_path








logo_path = 'PlanProLogo.png'
col1, col2, col3 = st.columns([1, 2, 1])  # Adjust column ratios for placement
with col2:  # Center the image in the middle column
    st.image(logo_path, width=150)  # Adjust width as needed