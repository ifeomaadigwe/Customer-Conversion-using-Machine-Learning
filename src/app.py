# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set Streamlit page config
st.set_page_config(page_title="Customer Conversion Dashboard", layout="wide")
st.title("Customer Conversion Dashboard")
st.markdown("Enter customer behavior data to predict conversion probability.")

# Use raw strings for Windows paths
MODEL_PATH = "src/customer_conversion_model.pkl"
SCALER_PATH = "src/scaler.pkl"

for path in [MODEL_PATH, SCALER_PATH]:
    if not os.path.exists(path):
        print(f"Path does not exist: {path}")
    else:
        print(f"Path exists: {path}")


# Load model and scaler with caching
@st.cache_resource(show_spinner=False)
def load_model_and_scaler():
    model = None
    scaler = None
    try:
        model = joblib.load(MODEL_PATH)
        st.success("Model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model: {e}")

    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
        except Exception as e:
            st.warning(f"⚠️ Error loading scaler: {e}")
    return model, scaler

model, scaler = load_model_and_scaler()

# Prediction function
def make_prediction(input_data):
    if model is None:
        return None, None
    if scaler is not None:
        input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
    return prediction[0], probability

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Model Predictions"])

if page == "Model Predictions":
    st.header("Model Predictions")
    st.write("Please provide the following customer data:")

    # Input fields
    col1, col2, col3 = st.columns(3)
    with col1:
        ads_clicks = st.number_input("Ads Clicks", min_value=0, max_value=10, value=5)
    with col2:
        time_on_site = st.number_input("Time on Site (minutes)", min_value=0, max_value=60, value=30)
    with col3:
        pages_visited = st.number_input("Pages Visited", min_value=1, max_value=20, value=10)

    # Predict button
    if st.button("Predict Conversion"):
        if model is None:
            st.error("The model isn't loaded. Check the model file path and format.")
        else:
            input_data = np.array([[ads_clicks, time_on_site, pages_visited]])
            prediction, probability = make_prediction(input_data)

            if prediction is None:
                st.error("Prediction could not be completed.")
            elif prediction == 1:
                st.success(f"✅ The customer is likely to convert! (Probability: {probability:.2f})")
            else:
                st.warning(f"⚠️ The customer is unlikely to convert. (Probability: {probability:.2f})")

    # Input guide
    st.markdown("---")
    st.subheader("Sample Input Guide")
    st.info("Typical values:\n- Ads Clicks (0-10)\n- Time on Site (0-60 minutes)\n- Pages Visited (1-20)")

    # Model information
    st.markdown("---")
    st.subheader("Model Information")
    st.write("* Model Type: Random Forest Classifier")
    st.write("* Features Used:")
    st.markdown("- Ads Clicks  \n- Time on Site  \n- Pages Visited")

    st.markdown("---")
    st.caption("Developed by Ifeoma Adigwe • Powered by Streamlit")
