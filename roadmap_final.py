import streamlit as st
st.set_page_config(layout="wide")
import requests
import folium
from streamlit_folium import st_folium
import polyline
import pandas as pd
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import google.generativeai as genai
import os
import re 

GOOGLE_API_KEY = "AIzaSyDKAflvuMl0OzaEQNYHyqssOgeR2mn4cis"
GEMINI_API_KEY = "AIzaSyBCzazAGe01lL0TNfTJUb_zmKRtRfgsupQ"

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-pro')

st.title("🗺 AI-Powered Route Planner with Accident Risk Analysis")

# File path constants
ACCIDENT_DATA_FILE = "accident_data_new.csv"
TRAINING_DATA_FILE = "accident_data_train.csv"

# Initialize session state variables if they don't exist
if "show_routes" not in st.session_state:
    st.session_state.show_routes = False
if "adding_incident" not in st.session_state:
    st.session_state.adding_incident = False
if "map_data" not in st.session_state:
    st.session_state.map_data = None
if "incident_added" not in st.session_state:
    st.session_state.incident_added = False

# Inputs
with st.container():
    st.subheader("📍 Enter Your Route Information")
    col1, col2 = st.columns(2)
    with col1:
        start = st.text_input("Starting Location", "Asia Pacific University", key="start_location")
        # Check if start location was updated
        if start != st.session_state.get("start", ""):
            st.session_state.start = start
            st.session_state.show_routes = True
            st.session_state.adding_incident = False

    with col2:
        end = st.text_input("Destination", "Pavilion Bukit Jalil", key="end_location")
        # Check if end location was updated
        if end != st.session_state.get("end", ""):
            st.session_state.end = end
            st.session_state.show_routes = True
            st.session_state.adding_incident = False

# Load Data
train_df = pd.read_csv(TRAINING_DATA_FILE)
predict_df = pd.read_csv(ACCIDENT_DATA_FILE)

# If an incident was just added, indicate that with a success message
if st.session_state.get("incident_added", False):
    st.success("✅ New incident has been added to the map and saved to the CSV file! Risk calculations have been updated.")
    st.session_state.incident_added = False  # Reset the flag

# Preview
with st.expander("📊 Accident Data Preview", expanded=False):
    st.dataframe(predict_df)

# Train Model
features = ["accident_type", "distance_to_route", "weather"]
target = "risk_index"

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['accident_type', 'weather']),
], remainder='passthrough')

risk_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train = train_df[features]
y_train = train_df[target]
risk_model.fit(X_train, y_train)

# Helpers
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in kilometers
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def get_coordinates(location):
    res = requests.get(f"https://maps.googleapis.com/maps/api/geocode/json?address={location}&key={GOOGLE_API_KEY}")
    if res.status_code == 200 and res.json()["status"] == "OK":
        loc = res.json()["results"][0]["geometry"]["location"]
        return loc["lat"], loc["lng"]
    return None, None

def get_weather(lat, lon):
    OPENWEATHER_API_KEY = "80e2abdd7a86e801165ed68bfe2acd31"
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    res = requests.get(url)
    if res.status_code == 200:
        weather_data = res.json()
        temperature = weather_data["main"]["temp"]
        feels_like = weather_data["main"]["feels_like"]
        windspeed = weather_data["wind"]["speed"]
        humidity = weather_data["main"]["humidity"]
        pressure = weather_data["main"]["pressure"]
        visibility = weather_data.get("visibility", "N/A")
        condition = weather_data["weather"][0]["description"].capitalize()
        weather_info = (
            f"🌡 Temperature: {temperature}°C (Feels like {feels_like}°C)\n"
            f"🌥 Condition: {condition}\n"
            f"💧 Humidity: {humidity}%\n"
            f"🌬 Wind Speed: {windspeed} m/s\n"
            f"🔵 Pressure: {pressure} hPa\n"
            f"👁 Visibility: {visibility/1000} km"
        )
        return weather_info
    return "Weather unavailable"

def extract_weather_condition(condition):
    # Helper to detect simple keywords
    for word in ["rain", "storm", "thunder", "flood"]:
        if word in condition.lower():
            return word
    return "normal"

def save_incident_to_csv(new_incident, file_path=ACCIDENT_DATA_FILE):
    """Save a new incident to the CSV file"""
    try:
        # Create a single-row dataframe for the new incident
        new_row_df = pd.DataFrame([new_incident])
        
        # Append to the CSV file without writing the header again
        with open(file_path, 'a') as f:
            new_row_df.to_csv(f, header=False, index=False)
        
        return True
    except Exception as e:
        st.error(f"Error saving incident to CSV: {e}")
        return False

# Button
if st.button("Show Routes"):
    st.session_state["start"], st.session_state["end"] = start, end
    st.session_state["show_routes"] = True
    st.session_state["adding_incident"] = False  # Reset incident adding mode

# Add Incident Button
if st.button("➕ Add Incident on Map"):
    if st.session_state.get("show_routes", False):
        st.session_state["adding_incident"] = True
    else:
        st.warning("Please show routes first before adding an incident.")

# Route + Risk Analysis
if st.session_state.get("show_routes", False):
    start, end = st.session_state["start"], st.session_state["end"]
    res = requests.get(f"https://maps.googleapis.com/maps/api/directions/json?origin={start}&destination={end}&alternatives=true&key={GOOGLE_API_KEY}")
    data = res.json()

    if data["status"] == "OK":
        routes = data["routes"]
        route_options = [f"Route {i+1}: {r['legs'][0]['distance']['text']} - {r['legs'][0]['duration']['text']}" for i, r in enumerate(routes)]
        selected_index = st.selectbox("Choose route to highlight:", range(len(routes)), format_func=lambda i: route_options[i])

        # Get destination weather FIRST
        lat, lng = get_coordinates(end)
        destination_weather = ""
        extracted_condition = "normal"
        if lat and lng:
            destination_weather = get_weather(lat, lng)
            extracted_condition = extract_weather_condition(destination_weather)

        boost_factor = 0.5
        try:
            gemini_prompt = (
                f"The weather at the destination is described as: '{destination_weather}'. "
                "Given this condition, what risk increase factor would you suggest for natural disaster accidents? "
                "Please reply with a single number like 1.0 (meaning 200% higher risk), assuming normal weather is 0.5"
            )
            gemini_response = gemini_model.generate_content(gemini_prompt)
            response_text = gemini_response.text.strip()

            # Extract only the first float number found in the text
            match = re.search(r"[-+]?\d*\.\d+|\d+", response_text)
            if match:
                boost_factor = float(match.group())
            else:
                raise ValueError(f"No number found in Gemini response: {response_text}")

        except Exception as e:
            st.warning(f"Gemini boost suggestion failed: {e}. Defaulting to 1.5x for rain/storms.")
            if extracted_condition != "normal":
                boost_factor = 1.0

        # Initialize map
        selected_coords = polyline.decode(routes[selected_index]["overview_polyline"]["points"])
        route_map = folium.Map(location=selected_coords[0], zoom_start=12)

        # Calculate real distances for all accident points in predict_df
        # This ensures newly added incidents have accurate distances too
        for i in range(len(predict_df)):
            acc_lat, acc_lng = predict_df.loc[i, "latitude"], predict_df.loc[i, "longitude"]
            
            # Add visual marker for all incidents on the map
            folium.Marker(
                [acc_lat, acc_lng], 
                tooltip=f"Incident: {predict_df.loc[i, 'accident_type']}",
                icon=folium.Icon(color="red", icon="warning-sign")
            ).add_to(route_map)
            
            # Calculate minimum distance to all routes
            min_distance = float('inf')
            for route_idx, route in enumerate(routes):
                coords = polyline.decode(route["overview_polyline"]["points"])
                for coord_lat, coord_lng in coords:
                    dist = haversine(acc_lat, acc_lng, coord_lat, coord_lng)
                    min_distance = min(min_distance, dist)
            
            # Update the distance in the dataframe
            predict_df.loc[i, "distance_to_route"] = min_distance

        risk_index = []
        accident_details = []

        # Process each route and calculate risk
        for i, route in enumerate(routes):
            coords = polyline.decode(route["overview_polyline"]["points"])
            route_risk = 0
            route_affected_accidents = 0

            for idx, row in predict_df.iterrows():
                acc_lat, acc_lng = row["latitude"], row["longitude"]
                
                # Calculate minimum distance to this specific route
                min_dist = min(haversine(acc_lat, acc_lng, lat, lng) for lat, lng in coords)

                # Only consider accidents within 0.5 km of the route
                if min_dist <= 0.5:
                    route_affected_accidents += 1
                    input_row = pd.DataFrame([{
                        "accident_type": row["accident_type"],
                        "distance_to_route": min_dist,
                        "weather": row["weather"]
                    }])
                    pred_risk = risk_model.predict(input_row)[0]

                    # Apply dynamic boost for weather conditions
                    if 'natural disaster' in row["accident_type"].lower():
                        pred_risk *= boost_factor

                    route_risk += pred_risk

                    accident_details.append({
                        "Route": i + 1,
                        "Type": row["accident_type"],
                        "Predicted Risk": round(pred_risk, 2),
                        "Distance to Route": round(min_dist, 3)
                    })

            # Factor in the total number of accidents near the route
            accident_multiplier = 1 + (route_affected_accidents * 0.1)
            avg_risk = (route_risk / max(route_affected_accidents, 1)) * accident_multiplier

            risk_index.append(avg_risk)

            # Color logic based on risk level
            color = "red" if avg_risk > 80 else "orange" if avg_risk > 50 else "yellow" if avg_risk > 30 else "green"
            if i == selected_index:
                color = "blue"
            
            # Add route line to map with risk information
            folium.PolyLine(
                coords, 
                color=color, 
                weight=5, 
                opacity=0.8,
                tooltip=f"Route {i+1}: Risk {avg_risk:.2f}"
            ).add_to(route_map)

            # Start and End marker for selected route
            if i == selected_index:
                folium.Marker(coords[0], tooltip="Start", icon=folium.Icon(color="green")).add_to(route_map)
                folium.Marker(coords[-1], tooltip="End", icon=folium.Icon(color="red")).add_to(route_map)

        route_map.add_child(folium.LatLngPopup())

        # Display the map
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Use single map display with returned objects only if adding an incident
            clicked = st_folium(route_map, width=800, height=600)

        # Display risk index and accident details
        if len(accident_details) > 0:
            st.subheader("📝 Accident Details")

            for i in range(len(accident_details)):
                acc_detail = accident_details[i]
                st.markdown(f"**Route {acc_detail['Route']}**")
                st.write(f"**Accident Type**: {acc_detail['Type']}")
                st.write(f"**Predicted Risk**: {acc_detail['Predicted Risk']}")
                st.write(f"**Distance to Route**: {acc_detail['Distance to Route']} km")
                st.write("-------")

                    # Weather
       # Weather
        if destination_weather:
            st.subheader("**🌤 Weather at Destination:**")
            st.markdown(destination_weather.replace("\n", "<br><br>"), unsafe_allow_html=True)
            st.write("-------")



        # Gemini risk advice
        try:
            selected_risk = risk_index[selected_index]
            gemini_prompt = f"You are a road safety expert. Analyze this route with a risk score of {selected_risk:.2f} out of 100. Provide safety insights and tips for the driver."
            gemini_response = gemini_model.generate_content(gemini_prompt)
            st.subheader("🧠 Gemini AI Risk Analysis")
            st.markdown(gemini_response.text)
        except Exception as e:
            st.error(f"Gemini API Error: {e}")
    else:
        st.error("Failed to fetch directions. Please check your inputs or API key.")
        
# Incident Addition
    if st.session_state.get("adding_incident", False):
        st.subheader("➕ Add Incident to Map")

        accident_type = st.selectbox("Select Accident Type", ["Normal Accident", "Natural Disaster"])
        lat = st.number_input("Latitude", value=3.0736, format="%.4f")
        lng = st.number_input("Longitude", value=101.6889, format="%.4f")
        weather = st.text_input("Weather Condition (e.g., rain, normal, etc.)", "normal")

        add_incident_button = st.button("Add Incident")
        
        if add_incident_button:
            new_incident = {
                "accident_type": accident_type,
                "latitude": lat,
                "longitude": lng,
                "weather": weather,
                
            }
            
            # Save the new incident
            if save_incident_to_csv(new_incident):
                st.session_state.incident_added = True

            
