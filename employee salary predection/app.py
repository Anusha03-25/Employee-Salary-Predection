import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("model.pkl")

st.title("🧑‍💼 Employee Salary Prediction App")

experience = st.number_input("Years of Experience", 0, 50)
education = st.selectbox("Education Level", ["Bachelors", "Masters", "PhD"])
role = st.selectbox("Job Role", ["Software Developer", "Web Developer", "Data Scientist", "Machine Learning Engineer"])

if st.button("Predict Salary"):
    input_data = pd.DataFrame([[experience, education, role]], columns=["experience", "education", "role"])
    salary = model.predict(input_data)[0]
    st.success(f"Predicted Annual Salary: ₹ {int(salary):,}")
