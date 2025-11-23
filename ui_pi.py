
import streamlit as st
import joblib
import numpy as np
import RPi.GPIO as GPIO
import time

# GPIO setup
RED_LED = 17
YELLOW_LED = 27
GREEN_LED = 22

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(RED_LED, GPIO.OUT)
GPIO.setup(YELLOW_LED, GPIO.OUT)
GPIO.setup(GREEN_LED, GPIO.OUT)

def turn_on_led(dementia_level):
    GPIO.output(RED_LED, dementia_level == "High")
    GPIO.output(YELLOW_LED, dementia_level == "Mild")
    GPIO.output(GREEN_LED, dementia_level == "Low")

# Load model and scaler
model = joblib.load("/mnt/data/dementia_project_rpi/tools/model_joblib")
scaler = joblib.load("/mnt/data/dementia_project_rpi/tools/scaler_joblib")

# Streamlit UI
st.title("Dementia Detection (with LED Alert)")

age = st.number_input("Age", min_value=0, max_value=120, step=1)
education = st.number_input("Years of Education", min_value=0, max_value=20, step=1)
ses = st.selectbox("Socioeconomic Status", [1, 2, 3, 4, 5])
mmse = st.slider("MMSE Score", 0.0, 30.0, 25.0)

if st.button("Predict"):
    features = np.array([[age, education, ses, mmse]])
    features_scaled = scaler.transform(features)
    result = model.predict(features_scaled)

    if result[0] == 2:
        st.error("High risk of Dementia")
        turn_on_led("High")
    elif result[0] == 1:
        st.warning("Mild Dementia")
        turn_on_led("Mild")
    else:
        st.success("Low or No Dementia")
        turn_on_led("Low")
