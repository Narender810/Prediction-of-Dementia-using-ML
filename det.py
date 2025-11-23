import streamlit as st
import pandas as pd
import joblib
import json
from streamlit_lottie import st_lottie
import time

def load_tools():
    scaler = joblib.load("./tools/scaler_joblib")
    model = joblib.load("./tools/model_joblib")
    return scaler, model

def predict_dementia(data, scaler, model):
    numerical_scale = ['MR Delay', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV']
    test = pd.DataFrame([data])
    test[numerical_scale] = scaler.transform(test[numerical_scale])
    confidence_score = model.predict_proba(test)[0][1]  
    prediction = model.predict(test)[0]  
    return prediction, confidence_score

def load_recommendations_from_json(file_path):
    with open(file_path, "r") as f:
        recommendations = json.load(f)
    return recommendations["dementia_recommendations"]  

def display_recommendations(recommendations):
    st.title("Recommendations")
    for category in recommendations:
        st.subheader(category["category"])
        for recommendation in category["recommendations"]:
            col1, col2 = st.columns([1, 5])
            with col1:
                st.image("./static/logo.png", width=130)
            with col2:
                st.markdown(f"**{recommendation['title']}**: {recommendation['description']}")
                st.write(f"*Details:* {recommendation['details']}")
                st.write(f"Video: [{recommendation['title']}]({recommendation['video_url']})")
def detection():
    scaler, model = load_tools()
    st.title("Dementia Detection")
    st.sidebar.header("Navigation")
    st.write("Please enter the patient's information:")
    mr_delay = st.number_input("MR Delay", key="mr_delay")
    gender = st.radio("Gender", ["Male", "Female"], key="gender")
    age = st.number_input("Age", key="age", step=1, format="%d")
    educ = st.number_input("Years of Education", key="educ", step=1, format="%d")
    ses = st.slider("Socioeconomic Status", min_value=1, max_value=5, key="ses")
    mmse = st.slider("MMSE Score", min_value=0, max_value=30, key="mmse")
    etiv = st.number_input("Estimated Total Intracranial Volume (cm³)", key="etiv")
    nwbv = st.number_input("Normalized Whole Brain Volume", key="nwbv")
    gender_numeric = 1 if gender == "Male" else 0

    if st.button("Predict"):
        st.markdown("---")

        
        with st.spinner('Please Wait We Are Getting the Best Results!'):
            time.sleep(3)
            data = {
                "MR Delay": mr_delay,
                "M/F": gender_numeric,
                "Age": age,
                "EDUC": educ,
                "SES": ses,
                "MMSE": mmse,
                "eTIV": etiv,
                "nWBV": nwbv
            }
            prediction, confidence_score = predict_dementia(data, scaler, model)
            if prediction == 1:
                st.error(f"The patient is predicted to have dementia with a confidence score of {confidence_score:.2f}.")
                st.markdown("---")

                st.info("Here are some recommendations to help manage dementia:")
                st.markdown("---")
                recommendations = load_recommendations_from_json("./knowledge_base/treat.json")

                display_recommendations(recommendations)
            else:
                st.success(f"The patient is predicted to not have dementia with a confidence score of {1 - confidence_score:.2f}.")

