from google_crc32c import value
import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import joblib


st.title("Austicm Spectrum Disorder Prediction")
st.write("This app predicts whether a person has Autism Spectrum Disorder (ASD) based on various features.")
st.write("Please fill in the following details to get the prediction:")

# Load the pre-trained model

model = load_model("asd_model.keras")


#load the label encoder
sc = joblib.load('scaler.pkl')

# Input fields for user data
#Index(['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score',
    #   'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'age', 'gender',
     #  'jundice', 'austim'],
     # dtype='object')

A1_Score = st.selectbox("A1 Score (0 or 1)", [0, 1])
A2_Score = st.selectbox("A2 Score (0 or 1)", [0, 1])
A3_Score = st.selectbox("A3 Score (0 or 1)", [0, 1])
A4_Score = st.selectbox("A4 Score (0 or 1)", [0, 1])
A5_Score = st.selectbox("A5 Score (0 or 1)", [0, 1])
A6_Score = st.selectbox("A6 Score (0 or 1)", [0, 1])
A7_Score = st.selectbox("A7 Score (0 or 1)", [0, 1])
A8_Score = st.selectbox("A8 Score (0 or 1)", [0, 1])
A9_Score = st.selectbox("A9 Score (0 or 1)", [0, 1])
A10_Score = st.selectbox("A10 Score (0 or 1)", [0, 1])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
gender = st.selectbox("Gender", ["Male", "Female"])
jundice = st.selectbox("Jaundice (0 or 1)", [0, 1])
austim = st.selectbox("Family History of Autism (0 or 1)", [0, 1])

# Prepare the input data for prediction
age_scaled = sc.transform([[age]])[0][0]  
gender = 1 if gender == 'Male' else 0

input_data = np.array([[A1_Score, A2_Score, A3_Score, A4_Score, A5_Score, A6_Score,
                        A7_Score, A8_Score, A9_Score, A10_Score, age_scaled, gender, jundice, austim]])
# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)
    if prediction[0][0] > 0.5:
        st.write("The model predicts that the person has Autism Spectrum Disorder (ASD).")
    else:
        st.write("The model predicts that the person does not have Autism Spectrum Disorder (ASD).")


