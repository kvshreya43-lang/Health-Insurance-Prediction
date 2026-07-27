import pandas as pd
import pickle
import streamlit as st

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("Medical Insurance Cost Predictor")

# Inputs
age = st.number_input("Age", 18, 100, 30)
sex = st.selectbox("Gender", ["Male", "Female"])
bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
children = st.number_input("Children", 0, 10, 0)
smoker = st.selectbox("Smoker", ["No", "Yes"])
region = st.selectbox(
    "Region",
    ["northeast", "northwest", "southeast", "southwest"]
)

if st.button("Predict Insurance Cost"):

    # BMI Category
    if bmi < 18.5:
        under = 1
        over = 0
        obese = 0
    elif bmi < 25:
        under = 0
        over = 0
        obese = 0
    elif bmi < 30:
        under = 0
        over = 1
        obese = 0
    else:
        under = 0
        over = 0
        obese = 1

    sex_female = 1 if sex == "Female" else 0
    sex_male = 1 if sex == "Male" else 0

    smoker_no = 1 if smoker == "No" else 0
    smoker_yes = 1 if smoker == "Yes" else 0

    region_northeast = 1 if region == "northeast" else 0
    region_northwest = 1 if region == "northwest" else 0
    region_southeast = 1 if region == "southeast" else 0
    region_southwest = 1 if region == "southwest" else 0

    smoker_bmi = smoker_yes * bmi

    sample = pd.DataFrame({
        "age":[age],
        "bmi":[bmi],
        "children":[children],

        "sex_female":[sex_female],
        "sex_male":[sex_male],

        "smoker_no":[smoker_no],
        "smoker_yes":[smoker_yes],

        "region_northeast":[region_northeast],
        "region_northwest":[region_northwest],
        "region_southeast":[region_southeast],
        "region_southwest":[region_southwest],

        "bmi_category_Obese":[obese],
        "bmi_category_Overweight":[over],
        "bmi_category_Underweight":[under],

        "smoker_bmi":[smoker_bmi]
    })

    prediction = model.predict(sample)

    st.success(f"Estimated Insurance Cost: ${prediction[0]:,.2f}")