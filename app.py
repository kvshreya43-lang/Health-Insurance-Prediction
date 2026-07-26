import streamlit as st
import pandas as pd
import pickle

# Load Trained Model
model = pickle.load(open("model.pkl", "rb"))

# Page Configuration
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="💰",
    layout="centered"
)

st.title("🏥 Medical Insurance Cost Prediction")
st.write("Predict your medical insurance charges using Linear Regression.")

st.markdown("---")

# User Inputs
age = st.slider("Age", 18, 64, 25)

sex = st.selectbox(
    "Gender",
    ["male", "female"]
)

bmi = st.number_input("BMI", 10.0, 60.0, 25.0)

children = st.selectbox(
    "Children",
    [0,1,2,3,4,5]
)

smoker = st.selectbox(
    "Smoker",
    ["yes","no"]
)

region = st.selectbox(
    "Region",
    ["northeast","northwest","southeast","southwest"]
)
input_data = pd.DataFrame({
    "age": [age],
    "bmi": [bmi],
    "children": [children],

    "sex_female": [1 if sex == "female" else 0],
    "sex_male": [1 if sex == "male" else 0],

    "smoker_no": [1 if smoker == "no" else 0],
    "smoker_yes": [1 if smoker == "yes" else 0],

    "region_northeast": [1 if region == "northeast" else 0],
    "region_northwest": [1 if region == "northwest" else 0],
    "region_southeast": [1 if region == "southeast" else 0],
    "region_southwest": [1 if region == "southwest" else 0]
})

if st.button("Predict Insurance Cost"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Insurance Charges: ${prediction[0]:,.2f}")

st.markdown("---")
st.caption("Developed using Python, Scikit-Learn, Streamlit & Linear Regression")