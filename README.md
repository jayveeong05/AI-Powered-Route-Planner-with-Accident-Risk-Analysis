[README.md](https://github.com/user-attachments/files/21651236/README.md)

# 🧭 AI-Powered Route Planner with Accident Risk Analysis

**Developer:** Ong Jyong Vey  
**Tech Stack:** Python, Streamlit, Google Maps API, OpenWeatherMap API, Gemini AI, Scikit-learn, Folium, Pandas

> An interactive web application that intelligently suggests the safest route between two locations using real-time weather, accident history, and machine learning-based risk analysis.

---

## 📌 Project Overview

This project was built as part of my exploration into smart urban navigation systems, combining **AI, geospatial data, and real-time APIs** to assist users in making safer travel decisions.

Unlike typical route planners, this application **prioritizes safety**, dynamically analyzing accident risks along different route options.

---

## 🎯 Key Objectives

- ✅ Fetch and display multiple travel routes using **Google Directions API**
- ✅ Integrate **historical accident data** and allow users to add new incidents dynamically
- ✅ Predict **accident risk scores** for routes using a trained **Random Forest Regressor**
- ✅ Adjust predictions based on **real-time weather conditions** using **OpenWeatherMap API**
- ✅ Boost risk prediction accuracy using contextual prompts with **Gemini AI**

---

## 💡 Features Snapshot

| Feature | Description |
|--------|-------------|
| 🗺 Route Suggestions | Multiple paths from source to destination |
| ⚠️ Accident Risk Markers | Accidents visualized along nearby roads |
| 🧠 AI-Based Risk Prediction | Random Forest model trained on real data |
| 🌦 Weather-Aware Adjustment | Risk scores vary based on current weather |
| 📍 Add Incident Points | Users can simulate incidents and see effects |
| 📊 Dynamic Risk Summary | Route-level summaries & per-incident details |

---

## 🖼 Demo Preview

> *Try routing from “Asia Pacific University” to “Pavilion Bukit Jalil” to see full system in action.*

| ![Map UI with Risks](https://via.placeholder.com/600x300?text=Map+View+with+Route+and+Risks) |  
|:--:|
| *Color-coded routes + markers = safety visualization* |

---

## 🛠 Technologies Used

| Category | Tools & Libraries |
|---------|-------------------|
| Frontend | Streamlit, Folium |
| Backend | Python, Pandas, Scikit-learn |
| APIs | Google Maps, OpenWeatherMap, Gemini AI |
| ML Model | RandomForestRegressor (Scikit-learn) |
| Others | Geopy, Requests, Joblib, Pickle |

---

## 🧪 ML Model: Risk Index Prediction

The system uses a supervised model to predict the **accident risk index** based on:

- Distance to route
- Type of accident
- Local weather conditions

Boosting logic is applied using Gemini's output when accident types like "natural disaster" are detected in hazardous weather scenarios.

---

## 📂 Folder Structure

```
📁 AI-Powered-Route-Planner
├── roadmap_final.py        # Main Streamlit app
├── accident_data_train.csv # Training data
├── accident_data_new.csv   # Added/combined incident data
├── requirements.txt        # Dependency list
├── .env                    # API keys (local setup)
```

---

## 🧑‍💻 How to Run Locally

1. **Clone this repo**
   ```bash
   git clone https://github.com/jayveeong05/AI-Powered-Route-Planner-with-Accident-Risk-Analysis.git
   cd AI-Powered-Route-Planner-with-Accident-Risk-Analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create `.env` file**
   ```bash
   GOOGLE_API_KEY=your_google_maps_api_key
   OPENWEATHER_API_KEY=your_openweather_api_key
   GEMINI_API_KEY=your_gemini_api_key
   ```

4. **Run the app**
   ```bash
   streamlit run roadmap_final.py
   ```

---

## 📈 Challenges Overcome

- Implemented **route-level distance-based accident filtering** using geospatial libraries
- Tuned a **customizable ML model** to handle both static and user-generated data
- Integrated **multi-API systems** with authentication and fallbacks
- Added a **visual dashboard** with meaningful color-coding for non-technical users

---

## 🚀 Future Plans

- 📍 Live traffic integration via Google Traffic Layer
- 🧠 Switch to XGBoost or ensemble methods for higher accuracy
- 📱 Responsive mobile layout with map interactivity
- 🛰 Auto-update from national traffic/accident databases

---

## 📬 Let's Connect!

If you're interested in how AI can improve transportation safety, feel free to reach out or explore the repo:

📧 [ongjyongvey@gmail.com](mailto:ongjyongvey@gmail.com)  
🔗 [LinkedIn Profile](https://linkedin.com/in/jyongvey) *(Insert your real link)*

---

## 📜 License

MIT License. You are free to use or extend this project with attribution.
