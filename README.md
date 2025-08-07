
# 🧭 Route Planner with Accident Risk Analysis

**Developer:** Jayvee  
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
```

---

## 📈 Challenges Overcome

- Implemented **route-level distance-based accident filtering** using geospatial libraries
- Tuned a **customizable ML model** to handle both static and user-generated data
- Integrated **multi-API systems** with authentication and fallbacks
- Added a **visual dashboard** with meaningful color-coding for non-technical users

---
