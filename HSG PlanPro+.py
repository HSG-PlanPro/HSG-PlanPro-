# This is the main page of the app. The App is set up as a multipage app. To set up the framework for the pages, the streamlit documentation for multipage apps was used: https://docs.streamlit.io/develop/concepts/multipage-apps/overview
# The project is stored publicly on Github as "HSG-PlanPro" https://github.com/HSG-PlanPro/HSG-PlanPro_Plus/blob/main/HSG%20PlanPro%2B.py
# Through discussion with our coach, we decided to put the inputs of Personal Information in the first page for ease of use.
# The API component was set up with the Microsoft Graph API, which required us to register an app. The correct API permissions were granted, most importantly Calendars.ReadWrite and Mail.Send
# The whole Azure OAuth Process was done with assistance of Chat-GPT. We created a
# The machine learning models were trained in Jupyter Notebooks using csv files. It is all archived in the same directory as the app in the 'Machine Learning Notebooks' folder.
# To train the ML Models, 2 main databases were used. Sleep quality information was retrieved from Kaggle: https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset?resource=download
# To train for day ratings (leisure and performance), day distributions were randomnly generated (100 days) and manually rated for both ratings.

# It collects user information (e.g., sleep data, personal details) and provides insights into time management and sleep quality.
# Users interact with this page to input personal details and receive initial recommendations on sleep habits.
import streamlit as st
import pandas as pd # Used for handling tabular data like sleep hours.
from datetime import datetime, timedelta # Handles date and time operations
import statistics as stat
import joblib # Loads the machine learning model for predicting sleep quality.
import numpy as np # Prepares numerical input for ML models as an array.
import matplotlib.pyplot as plt # Visualizes data using bar charts.
import tempfile # Creates temporary files for storing the bar charts to be reused in the PDF report.

st.set_page_config(
    page_title="HSG PlanPro+",
    page_icon="🗓 - Scheduler",
    layout='wide'      # Sets up the Web App to use the full screen on computers for streamlit instead of the narrow standard display.
)

st.title('HSG PlanPro+')
st.subheader('The Smart Scheduling Assistant for HSG Students')

st.write('Many students struggle with organizing their time. We believe that visualizing personal organization and time management goes a long way in building a mental framework of how time is used.')
         
st.write('This is why we developped PlanPro+. Our software allows you to interact with your Outlook calendar and provides you with multiple visual representations of your calendar.')
         
st.write(""""What's new?" you ask. Well, we allow you to define a timeframe and visualize how your time is allocated within it and we show you your sleep schedule.""")
         
st.write('But most importantly, we study your time organization and your sleep and use machine learning models to make recommendations about **how you could optimize your life**. Hooked?')

DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
today = datetime.today()  #set today's date

def calculate_sleep_deficit(age, avg_sleep):
    """
    Calculates the sleep deficit based on the user's age and average sleep duration. They were found in some basic internet research to be the biggest factors.
    The optimal sleeping times are retrieved from https://www.mayoclinic.org/healthy-lifestyle/adult-health/expert-answers/how-many-hours-of-sleep-are-enough/faq-20057898
    This is not Machine Learning, the sleep quality predictor is implemented later.

    Parameters:
    - age: int, user's age in years.
    - avg_sleep: float, average hours of sleep per night.

    Returns:
    - A string message recommending whether the user should increase their sleep duration.
    """
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

    # Create columns for each day of the week, using the DAYS list for correct day order
    cols = st.columns(len(DAYS))
    
    # Input fields for each day in separate columns
    # Create a dictionary to collect sleep hours for each day of the week.
    # Each day corresponds to a column in the Streamlit layout, and the user inputs their sleep hours in the number_input fields.
    # - Keys: Days of the week (e.g., "Monday", "Tuesday").
    # - Values: User-entered sleep hours (restricted between 0 and 24).
    sleep_hours = {day: cols[i].number_input(day, min_value=0, max_value=24) for i, day in enumerate(DAYS)}
    
    if st.form_submit_button("Analyze Sleep"):
        # Calculate statistics
        avg_sleep = stat.mean(sleep_hours.values())
        total_sleep = sum(sleep_hours.values())
        
        # Display results
        st.write(f"Welcome, {firstname}!")
        st.write(f"Average Sleep: {avg_sleep:.2f} hours")
        st.write(f"Total Weekly Sleep: {total_sleep} hours")
        st.write(calculate_sleep_deficit(datetime.now().year - date_of_birth.year, avg_sleep))
        deficit_message = calculate_sleep_deficit(datetime.now().year - date_of_birth.year, avg_sleep)

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

        # Machine learning part starts, hence the ml_ in front of variables
        ml_daily_steps = daily_steps
        ml_sleep_duration = stat.mean(sleep_hours.values())
        ml_age = today.year - date_of_birth.year - ((today.month, today.day) < (date_of_birth.month, date_of_birth.day))

        # Sleep Quality Model
        model = joblib.load('sleep_quality_model.pkl')

        # Prepare the input data use in the predictor
        input_data = np.array([[ml_age, ml_sleep_duration, ml_daily_steps]])
        prediction = model.predict(input_data)
        sleep_message = (f"Predicted Sleep Quality: {prediction[0]:.2f}") #here the returned value is not really understandable for the reader as the model is trained to return sleep qualities from 4 (least) to 9 (best)

        #variable saved into session_states
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

# Set the 'Day' column as categorical for correct ordering, might not be necessary anymore, but in classic developper fashion, if it ain't broke, don't fix it
sleep_df['Day'] = pd.Categorical(sleep_df['Day'], categories=DAYS, ordered=True)

# Plot bar chart using matplotlib
fig, ax = plt.subplots(figsize=(13, 6))  # Customize the figure size
ax.bar(sleep_df['Day'], sleep_df['Sleep Hours'], color='skyblue')

# Add titles and labels
ax.set_title("Sleep Hours Per Night", fontsize=14)
ax.set_xlabel("Day", fontsize=12)
ax.set_ylabel("Sleep Hours", fontsize=12)

# Create a temporary file to store the chart and use it in the PDF later through the session_state below
with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
    chart_image_path = temp_file.name
    fig.savefig(chart_image_path)

st.session_state['chart_image_path'] = chart_image_path

logo_path = 'PlanProLogo.png'
col1, col2, col3 = st.columns([1, 2, 1])  # Adjusted column ratios for placement
with col2:
    st.image(logo_path, width=150)  # Adjusted width