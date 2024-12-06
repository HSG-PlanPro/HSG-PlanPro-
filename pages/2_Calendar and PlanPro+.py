from streamlit_calendar import calendar
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, time
import statistics as stat
from msal import PublicClientApplication
import requests
import pytz
import matplotlib.pyplot as plt
import numpy as np
import joblib
from fpdf import FPDF
from io import BytesIO
import os
import tempfile
import base64

# Constants
CLIENT_ID = '245bdaad-a3d0-44ce-9d77-5fe830189ca1'
AUTHORITY = 'https://login.microsoftonline.com/common'
SCOPES = ['Calendars.ReadWrite', 'Mail.Send']
REDIRECT_URI = 'http://localhost'
LOCAL_TIMEZONE = pytz.timezone("Europe/Zurich")
DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Initialize session state variables
if "access_token" not in st.session_state:
    st.session_state["access_token"] = None
if "events" not in st.session_state:
    st.session_state["events"] = []

# Helper Functions
def authenticate():
    """Authenticate the user and retrieve an access token."""
    app = PublicClientApplication(CLIENT_ID, authority=AUTHORITY)
    result = app.acquire_token_interactive(scopes=SCOPES)
    return result.get('access_token')

def create_calendar_event(access_token, new_event):
    """Create a new calendar event."""
    url = 'https://graph.microsoft.com/v1.0/me/events'
    headers = {'Authorization': f'Bearer {access_token}', 'Content-Type': 'application/json'}
    response = requests.post(url, headers=headers, json=new_event)
    if response.status_code == 201:
        st.success(f"Event created successfully: {new_event['subject']}")
    else:
        st.error(f"Failed to create event: {response.json()}")

def convert_to_utc(local_time):
    """Convert local time to UTC."""
    return LOCAL_TIMEZONE.localize(local_time).astimezone(pytz.utc)

def format_event_details(event):
    """Format event details for display."""
    # Parse the start and end times as UTC
    start_time_utc = datetime.fromisoformat(event['start']['dateTime']).replace(tzinfo=pytz.utc)
    end_time_utc = datetime.fromisoformat(event['end']['dateTime']).replace(tzinfo=pytz.utc)

    # Convert to Swiss local time
    start_time = start_time_utc.astimezone(LOCAL_TIMEZONE)
    end_time = end_time_utc.astimezone(LOCAL_TIMEZONE)

    # Format location information
    location = event.get('location', {}).get('displayName', 'No Location')

    # Return formatted event details
    return f"- **{event['subject']}** ({location}) from {start_time:%d.%m.%Y %H:%M} to {end_time:%d.%m.%Y %H:%M}"


def calculate_total_hours(events, category, start_date, end_date):
    total_hours = 0
    
    for event in events:
        event_start = event.get('start', {}).get('dateTime', None)
        event_end = event.get('end', {}).get('dateTime', None)
        event_location = event.get('location', {}).get('displayName', None)

        # Ensure event_start and event_end are valid datetime objects
        if event_start and event_end:
            event_start = datetime.fromisoformat(event_start)
            event_end = datetime.fromisoformat(event_end)

            # Calculate duration in hours
            duration = (event_end - event_start).total_seconds() / 3600

            # Check if the event falls within the specified date range
            if start_date <= event_start.date() <= end_date:
                # Check if the category matches or if the location is defined
                if category is None or (event_location and category.lower() in event_location.lower()):
                    total_hours += duration

    return total_hours

def fetch_all_calendar_events(access_token, start_date, end_date):
    """Fetch all calendar events from all user calendars within a date range."""
    base_url = "https://graph.microsoft.com/v1.0"
    headers = {'Authorization': f'Bearer {access_token}'}
    all_events = []

    # Get all calendars
    calendar_url = f"{base_url}/me/calendars"
    calendar_response = requests.get(calendar_url, headers=headers)
    
    if calendar_response.status_code != 200:
        st.error(f"Failed to fetch calendars: {calendar_response.json()}")
        return []
    
    calendars = calendar_response.json().get('value', [])
    
    for calendar in calendars:
        calendar_id = calendar['id']
        events_url = f"{base_url}/me/calendars/{calendar_id}/events?$filter=start/dateTime ge '{start_date}' and end/dateTime le '{end_date}'"
        
        while events_url:
            response = requests.get(events_url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                all_events.extend(data.get('value', []))
                events_url = data.get('@odata.nextLink')  # Check for next page
            else:
                st.error(f"Failed to fetch events for calendar {calendar_id}: {response.json()}")
                break

    # Eliminate duplicate events
    unique_events = {(e['subject'].strip().lower(), e['start']['dateTime']): e for e in all_events}.values()
    return list(unique_events)

def filter_all_day_events(events):
    """Filter out all-day events from the events list."""
    filtered_events = [
        event for event in events if not event.get('isAllDay', False)
    ]
    return filtered_events

# Function to tally hours for a single day
def tally_hours(events):
    categories = {'Deadline': 0, 'Work': 0, 'Leisure/Break': 0, 'Chore': 0, 'School/Study': 0}
    for event in events:
        location = event.get('location', {}).get('displayName', '').strip()
        event_hours = (datetime.fromisoformat(event['end']['dateTime']) - datetime.fromisoformat(event['start']['dateTime'])).total_seconds() / 3600
        if location in categories:
            categories[location] += event_hours
        else:
            categories['School/Study'] += event_hours
    return categories

def generate_report(start_date, access_token):
    report_data = []  # To store the data for the report
    week_days = [start_date + timedelta(days=i) for i in range(8)]  # Create a list of the 8 days (including start date + 7 days)
    end_date = start_date + timedelta(days=7)
    all_events = fetch_all_calendar_events(access_token, start_date.isoformat(), end_date.isoformat())

    # Loop through each day of the 8-day period and process events
    for day in week_days:
        day_start = day.isoformat() + 'T00:00:00'
        day_end = day.isoformat() + 'T23:59:59'
        day_events = [event for event in all_events if event['start']['dateTime'] >= day_start and event['end']['dateTime'] <= day_end]

        # Define categories and tally hours
        category_hours = {'Deadline': 0, 'Work': 0, 'Leisure/Break': 0, 'Chore': 0, 'School/Study': 0}
        for event in day_events:
            location = event.get('location', {}).get('displayName', '').strip()
            event_hours = (datetime.fromisoformat(event['end']['dateTime']) - datetime.fromisoformat(event['start']['dateTime'])).total_seconds() / 3600
            if location in category_hours:
                category_hours[location] += event_hours
            else:
                if event_hours > 0:
                    category_hours['School/Study'] += event_hours

        # Adjust for free hours
        total_busy_hours = sum(category_hours.values())
        free_hours = max(0, 14 - total_busy_hours)
        category_hours['Leisure/Break'] += free_hours

        report_data.append({
            'date': day,
            'category_hours': category_hours,
            'performance_rating': None,  # Placeholder
            'leisure_rating': None,  # Placeholder
        })

    # Predict performance and leisure ratings
    performance_model = joblib.load('model_rf_performance.pkl')
    leisure_model = joblib.load('model_rf_leisure.pkl')
    for day_data in report_data:
        hours = day_data['category_hours']
        input_data = np.array([[hours['Work'], hours['School/Study'], hours['Leisure/Break'], hours['Chore']]])
        day_data['performance_rating'] = performance_model.predict(input_data)[0]
        day_data['leisure_rating'] = leisure_model.predict(input_data)[0]

    # Generate the PDF report

    

    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    logo_path = "PlanProLogo.png"  # Path to the logo file
    pdf.image(logo_path, x=175, y=8, w=25)  # Adjust `x`, `y`, and `w` as needed
    pdf.set_font("Times", style='B', size=16)
    pdf.cell(0, 10, txt=f"Weekly Report for {st.session_state["firstname"]} {st.session_state["lastname"]}", ln=True, align='L')
    pdf.set_font("Times", style='', size=12)
    pdf.multi_cell(160, 5, txt="The ratings are given based on a Machine Learning Model and range from Very Bad (0) to Very Good (3).", align='L')
    pdf.ln(5)

    # Counter to track entries and trigger page breaks
    day_counter = 0

    for day_data in report_data:
        # Increment the counter
        day_counter += 1

        text_start_y = pdf.get_y()
        # Day header
        pdf.set_font("Arial", size=10, style='B')
        pdf.cell(0, 10, f"{day_data['date'].strftime('%A %d.%m.%Y')}", ln=True)

        # Ratings and Comments
        if day_data['performance_rating'] >= day_data['leisure_rating']:
            comment1 = "This day's organization is performance oriented. Check if that is what you want. If yes, stay focused. "
            if day_data['performance_rating'] > 2:
                comment11 = 'You should have a very productive day, carry on. '
            elif 2 >= day_data['performance_rating'] > 1:
                comment11 = "You should have a productive day. Stay on top of your duties, but don't forget the rest of your life. "
            elif 1 >= day_data['performance_rating'] > 0:
                comment11 = 'This lifestyle is not sustainable. You are either working way too much or too little. '
        else:
            comment1 = 'This day is leisure oriented. Check if this is your current priority. If yes, relax and enjoy! '
            if day_data['leisure_rating'] > 2:
                comment11 = 'You should be able to relax and have fun while fulfilling your duties. Nice work! Do not get too lazy though! '
            elif 2 >= day_data['leisure_rating'] > 1:
                comment11 = "You're quite balanced. You could afford to rest a bit more. "
            elif 1 >= day_data['leisure_rating'] > 0:
                comment11 = 'If you want to rest, this is not it at all! '

        pdf.set_font("Arial", size=10)
        pdf.cell(50, 10, f"Performance Rating: {day_data['performance_rating']:.2f}", ln=0)
        pdf.cell(50, 10, f"Leisure Rating: {day_data['leisure_rating']:.2f}", ln=1)
        comments_text = f"Comments: {comment1} More precisely: {comment11}"
        pdf.multi_cell(115, 5, comments_text)
        pdf.ln(3)

        # Category Hours
        category_hours_filtered = {k: v for k, v in day_data['category_hours'].items() if v > 0}
        for category, hours in category_hours_filtered.items():
            pdf.set_font("Arial", size=8)
            pdf.cell(50, 5, f"      - {category}: {hours:.1f}h", ln=True)

        # Add Pie Chart
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(category_hours_filtered.values(), labels=category_hours_filtered.keys(), autopct='%1.1f%%', startangle=90)
        ax.axis('equal')

        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            fig.savefig(temp_file.name)
            temp_file_path = temp_file.name

        pdf.image(temp_file_path, x=130, y=text_start_y, w=55)
        os.remove(temp_file_path)

        pdf.ln(25)  # Add space for the next day's content

        # Add a page break every three days
        if day_counter % 3 == 0 and day_counter != len(report_data):  # Prevent extra page after the last day
            pdf.add_page()

    pdf.set_font("Times", style='B', size=16)
    pdf.cell(280, 10, txt="Sleep and General Health", ln=True, align='L')
    pdf.set_font("Times", style='', size=12)
    pdf.multi_cell(0, 5, txt="Based on your personal information, you will find a simple assessment of your sleep quality here. The evaluation is made with a Machine Learning Model that compares your sleep habits to those of many others.", align='L')
    text_start_y2 = pdf.get_y()
    pdf.ln(5)

    # Add table headers
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(40, 10, "Metric", border=1, align="L")
    pdf.cell(15, 10, "Value", border=1, ln=True, align="L")

    # Add table rows
    pdf.set_font("Arial", size=10)
    data = [
        ("Age", st.session_state["ml_age"]),
        ("Average Sleep (hrs)", f"{st.session_state['avg_sleep']:.2f}"),
        ("Average Daily Steps", st.session_state["daily_steps"]),
    ]

    for metric, value in data:
        pdf.cell(40, 10, metric, border=1)
        pdf.cell(15, 10, str(value), border=1, ln=True)

    # Check if the chart image exists in session_state
    if 'chart_image_path' in st.session_state:
        chart_image_path = st.session_state['chart_image_path']
        if os.path.exists(chart_image_path):
            pdf.image(chart_image_path, x=80, y=text_start_y2, w=120)  # Adjust position and size
            

    pdf.ln(5)
    pdf.multi_cell(0, 5, st.session_state['deficit_message'])

    sleep_predi = st.session_state['sleep_prediction']
    if sleep_predi < 6:
        advice = '''Your sleep is insufficient. Some steps you need to take to improve your sleep:
             - Stick to a regular sleep schedule.
             - Expose your eyes to direct sunlight for a couple of minutes in the morning to set up your Circadian 
               Rythm.
             - Increase your physical activity by walking more.
             - Organize yourself to increase the amount of sleep hours.
        '''
    elif 6 <= sleep_predi < 8:
        advice = '''Your sleep is quite good. Some steps you can take to improve your energy levels:
             - Maintain your current schedule.
             - Walk more or engage in other physical activity.
             - Organize yourself to be able to sleep a bit more. You could for example leave your phone outside of 
               your room.
        '''
    elif sleep_predi >= 8:
        advice = '''Your sleep is excellent. Don't spend your life in bed. Prioritize uninterrupted sleep nights that don't exceed 9 hours.
        '''

    pdf.ln(3)
    pdf.multi_cell(0, 5, txt=advice)

    # Save the PDF to a file
    pdf_output_path = 'weekly_report_updated.pdf'
    if os.path.exists(pdf_output_path):
        os.remove(pdf_output_path)
    pdf.output(pdf_output_path, "F")

    return pdf_output_path

def send_email_via_graph_api(report_file, recipient_email):
    """
    Send an email with the report attached via Microsoft Graph API.
    """
    # Read the PDF file in binary mode
    with open(report_file, "rb") as file:
        pdf_content = file.read()

    # Encode the PDF content in Base64
    encoded_pdf = base64.b64encode(pdf_content).decode("utf-8")

    # Construct the email payload
    email_payload = {
        "message": {
            "subject": "Weekly Report",
            "body": {
                "contentType": "Text",
                "content": "Please find the weekly report attached."
            },
            "toRecipients": [
                {"emailAddress": {"address": recipient_email}}
            ],
            "attachments": [
                {
                    "@odata.type": "#microsoft.graph.fileAttachment",
                    "name": os.path.basename(report_file),
                    "contentBytes": encoded_pdf
                }
            ]
        },
        "saveToSentItems": "true"
    }

    # API URL
    url = "https://graph.microsoft.com/v1.0/me/sendMail"

    # Headers
    headers = {
        "Authorization": f"Bearer {st.session_state['access_token']}",
        "Content-Type": "application/json"
    }

    # Send the email
    response = requests.post(url, headers=headers, json=email_payload)

    # Handle response
    if response.status_code == 202:
        return "Email sent successfully!"
    else:
        raise Exception(f"Failed to send email: {response.json()}")

st.title('Calendar View')
st.subheader('2. Click on *Synchronise with Outlook* to retrieve your calendar and get it on here.')
st.write('You will be redirected to a Microsoft page asking for authorization to access to your Outlook. Follow their instructions. **You have to use a private account. Your institution mail will not grant these permissions**.')
st.write('If your browser has preloaded the wrong Microsoft account, close the authentication window and press *Synchronise with Oulook* again.')
st.write('We recommend to first add your HSG Calendar from Compass to your personal Outlook Account. Once your calendar is retrieved, you can play with the calendar interface. The classic Outlook weekly view can be seen with the *timegrid* mode.')
# Calendar view selection

mode = st.selectbox(
    "Calendar Mode:",
    (
        "daygrid",
        "timegrid",
        "list",
    ),
)

calendar_options = {
    "editable": "true",
    "navLinks": "true",
    "selectable": "true",
}

# Modify calendar options based on selected view mode
if mode == "daygrid":
    calendar_options = {
        **calendar_options,
        "headerToolbar": {
            "left": "today prev,next",
            "center": "title",
            "right": "dayGridDay,dayGridWeek,dayGridMonth",
        },
        "initialDate": "2024-12-01",
        "initialView": "dayGridMonth",
    }
elif mode == "timegrid":
    calendar_options = {
        **calendar_options,
        "initialView": "timeGridWeek",
    }
elif mode == "list":
    calendar_options = {
        **calendar_options,
        "initialDate": "2024-11-01",
        "initialView": "listMonth",
    }

# Add a button to fetch events from Outlook calendar
if st.button("Synchronise with Outlook"):
    if st.session_state["access_token"] is None:
        st.error("You need to authenticate first!")
        st.session_state["access_token"] = authenticate()  # Trigger authentication process if needed
    
    if st.session_state["access_token"]:
        start_date = (datetime.now() - timedelta(days=100)).isoformat()
        end_date = (datetime.now() + timedelta(days=100)).isoformat()
        
        # Use the updated function to fetch events from all calendars
        events_from_outlook = fetch_all_calendar_events(st.session_state["access_token"], start_date, end_date)
        
        if events_from_outlook:
            # Filter out multi-day events
            single_day_events = filter_all_day_events(events_from_outlook)
            
            # Prepare events for the calendar
            events = [
                {
                    "title": event['subject'],
                    "color": "#71CFD8",
                    "start": (datetime.fromisoformat(event['start']['dateTime']) + timedelta(hours=1)).isoformat(),
                    "end": (datetime.fromisoformat(event['end']['dateTime']) + timedelta(hours=1)).isoformat(),
                }
                for event in single_day_events
            ]
            st.session_state["events"] = events
            st.success("Events successfully fetched and added to the calendar.")
        else:
            st.error("No events found for the selected timeframe.")

# Streamlit calendar widget
state = calendar(
    events=st.session_state.get("events", []),  # Use events from session state (empty list initially)
    options=calendar_options,
    custom_css="""
    .fc-event-past {
        opacity: 0.8;
    }
    .fc-event-time {
        font-style: italic; 
    }
    .fc-event-title {
        font-weight: 700;
    }
    .fc-toolbar-title {
        font-size: 2rem;
    }
    """,
    key=mode,
)

def main():

    # Calendar Integration
    st.subheader("3. You can create new events here. ")
    st.write("If your calendar is empty, input all your events here at once. You will need to restart HSG PlanPro+ for these events to be displayed. Events that are already in the Outlook Calendar will be labelled as school events, since it is expected that you already added your HSG-Calendar from Compass to your personal Outlook Calendar.")
    if "access_token" not in st.session_state:
        if st.button("Authenticate"):
            token = authenticate()
            if token:
                st.session_state["access_token"] = token
    else:
        
        access_token = st.session_state["access_token"]

        # Create Event
        with st.expander("Create New Event"):
            subject = st.text_input("Event")
            location = st.selectbox("Type", ('Deadline', 'Work', 'School/Study', 'Leisure/Break', 'Chore'))
            start_date = st.date_input("Start Date")
            start_time = st.time_input("Start Time")
            end_date = st.date_input("End Date")
            end_time = st.time_input("End Time")
            if st.button("Create Event"):
                new_event = {
                    "subject": subject,
                    "location": {"displayName": location},
                    "start": {"dateTime": convert_to_utc(datetime.combine(start_date, start_time)).isoformat(), "timeZone": "UTC"},
                    "end": {"dateTime": convert_to_utc(datetime.combine(end_date, end_time)).isoformat(), "timeZone": "UTC"}
                }
                create_calendar_event(access_token, new_event)

        # Pie Chart of tallied hours by category (type)
        st.subheader('4. (Optional) You can define a time frame and look at your time allocation by category.')
        with st.expander("Tally Hours by Category"):
            if "access_token" in st.session_state:
                access_token = st.session_state["access_token"]

                # Set the start and end dates for the time period
                tally_start_date = st.date_input("Tally Start Date", datetime.now().date())
                tally_end_date = st.date_input("Tally End Date", (datetime.now() + timedelta(days=30)).date())

                if st.button("Calculate Hours"):
                    events = fetch_all_calendar_events(access_token, tally_start_date.isoformat(), tally_end_date.isoformat())
                    if events:
                        # Define categories for tallying
                        predefined_categories = ['Deadline', 'Work', 'Leisure/Break', 'Chore', 'School/Study']
                        category_hours = {category: 0 for category in predefined_categories}

                        # Calculate total hours for all events
                        for event in events:
                            location = event.get('location', {}).get('displayName', '').strip()
                            event_hours = (datetime.fromisoformat(event['end']['dateTime']) - datetime.fromisoformat(event['start']['dateTime'])).total_seconds() / 3600

                            # Add the event hours to the category based on location
                            if location in predefined_categories:
                                category_hours[location] += event_hours
                            else:
                                if event_hours > 0:  # Only add to School/Study if there's a positive duration
                                    category_hours['School/Study'] += event_hours

                        # Calculate total busy hours
                        total_busy_hours = sum(category_hours.values())
                        total_available_hours = 14
                        free_hours = max(0, total_available_hours - total_busy_hours)

                        # Add free hours to leisure hours
                        category_hours['Leisure/Break'] += free_hours

                        # Filter out categories with 0 hours
                        category_hours_filtered = {k: v for k, v in category_hours.items() if v > 0}

                        if category_hours_filtered:
                            # Create a pie chart for the total hours (only for non-zero values)
                            fig, ax = plt.subplots(figsize=(3,3))
                            ax.pie(category_hours_filtered.values(), labels=category_hours_filtered.keys(), autopct='%1.1f%%', startangle=90)
                            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

                            # Display the pie chart
                            st.pyplot(fig)

                            # Optionally, show the exact total hours for each category
                            st.write("Total Hours by Category:")
                            st.write(category_hours_filtered)
                        else:
                            st.write("No hours logged for the selected categories in the specified timeframe.")
                    else:
                        st.write("No events found for the specified timeframe.")

        # Streamlit UI for Generating and Sending the Report
        st.subheader('5. Here you can generate a weekly report with recommendations and visualizations. You can download it or export the PDF to your e-mail.')
        with st.expander("Generate Weekly Reporting"):
            start_date = st.date_input("Select Start Date for the Week", datetime.now().date())

            # Track the report generation state
            if "report_generated" not in st.session_state:
                st.session_state["report_generated"] = False

            # Generate the report
            if st.button("Generate Weekly Report"):
                try:
                    # Generate the report and save it as a file
                    report_file = "weekly_report_updated.pdf"
                    generate_report(start_date, access_token)
                    st.session_state["report_generated"] = True  # Update state
                    st.success("Weekly report generated successfully!")

                    # Provide a download button for the generated report
                    with open(report_file, "rb") as file:
                        st.download_button(
                            label="Download Report",
                            data=file,
                            file_name="weekly_report_updated.pdf",
                            mime="application/pdf"
                        )

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

            # Email Input and Sending Logic
            if st.session_state["report_generated"]:
                recipient_email = st.text_input("Enter recipient's email address", placeholder="example@example.com")

                if recipient_email:  # Display the send button if an email is entered
                    if st.button("Send Report via Email"):
                        try:
                            email_status = send_email_via_graph_api("weekly_report_updated.pdf", recipient_email)
                            st.success(email_status)
                        except Exception as e:
                            st.error(f"Failed to send email: {str(e)}")

if __name__ == "__main__":
    main()