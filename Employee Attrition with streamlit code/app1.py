import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("model.pkl", "rb"))

st.title("Employee Attrition Prediction App")

Age = st.number_input("Age", 18, 60)
DailyRate = st.number_input("Daily Rate", 100, 1500)
MonthlyIncome = st.number_input("Monthly Income", 1000, 20000)
JobLevel = st.selectbox("Job Level", [1, 2, 3, 4, 5])
TotalWorkingYears = st.number_input("Total Working Years", 0, 40)

Gender = st.selectbox("Gender", ["Male", "Female"])
Department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
BusinessTravel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
EducationField = st.selectbox("Education Field",
                              ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])
MaritalStatus = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])

# Create DataFrame EXACTLY like model was trained
input_data = pd.DataFrame([{
    "Age": Age,
    "DailyRate": DailyRate,
    "MonthlyIncome": MonthlyIncome,
    "JobLevel": JobLevel,
    "Gender": Gender,
    "Department": Department,
    "BusinessTravel": BusinessTravel,
    "EducationField": EducationField,
    "MaritalStatus": MaritalStatus,
    "TotalWorkingYears": TotalWorkingYears
}])

if st.button("Predict"):
    result = model.predict(input_data)[0]

    if result == 1:
        st.error("⚠ Employee will likely leave.")
    else:
        st.success("✔ Employee will likely stay.")
