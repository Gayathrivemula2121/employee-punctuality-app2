# Now, the Streamlit app code.
# Save this as a separate Python file, e.g., app.py
# Run it with: streamlit run app.py

import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the saved model
with open('attrition_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the features (from the notebook)
numerical_cols = ['JobInvolvement', 'PerformanceRating', 'EnvironmentSatisfaction', 'JobSatisfaction', 
                  'WorkLifeBalance', 'Age', 'DistanceFromHome', 'Education', 'JobLevel', 'MonthlyIncome', 
                  'NumCompaniesWorked', 'PercentSalaryHike', 'StockOptionLevel', 'TotalWorkingYears', 
                  'TrainingTimesLastYear', 'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']

# Streamlit app title
st.title("Employee Attrition Prediction")

# Input fields for numerical features
st.header("Numerical Features")
num_inputs = {}
for col in numerical_cols:
    if col in ['JobInvolvement', 'PerformanceRating', 'EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance', 'Education', 'JobLevel', 'StockOptionLevel']:
        num_inputs[col] = st.number_input(col, min_value=1, max_value=5, value=3)  # Assuming rating scales 1-4 or 1-5
    elif col == 'Age':
        num_inputs[col] = st.number_input(col, min_value=18, max_value=100, value=30)
    elif col == 'DistanceFromHome':
        num_inputs[col] = st.number_input(col, min_value=1, max_value=30, value=5)
    elif col == 'MonthlyIncome':
        num_inputs[col] = st.number_input(col, min_value=1000, max_value=200000, value=5000)
    elif col == 'NumCompaniesWorked':
        num_inputs[col] = st.number_input(col, min_value=0, max_value=10, value=1)
    elif col == 'PercentSalaryHike':
        num_inputs[col] = st.number_input(col, min_value=0, max_value=30, value=10)
    elif col == 'TotalWorkingYears':
        num_inputs[col] = st.number_input(col, min_value=0, max_value=50, value=5)
    elif col == 'TrainingTimesLastYear':
        num_inputs[col] = st.number_input(col, min_value=0, max_value=10, value=2)
    elif col == 'YearsAtCompany':
        num_inputs[col] = st.number_input(col, min_value=0, max_value=50, value=3)
    elif col == 'YearsSinceLastPromotion':
        num_inputs[col] = st.number_input(col, min_value=0, max_value=20, value=1)
    elif col == 'YearsWithCurrManager':
        num_inputs[col] = st.number_input(col, min_value=0, max_value=20, value=2)

# Input fields for categorical features
st.header("Categorical Features")
cat_inputs = {}
cat_inputs['BusinessTravel'] = st.selectbox('BusinessTravel', options=['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'])
cat_inputs['Department'] = st.selectbox('Department', options=['Sales', 'Research & Development', 'Human Resources'])
cat_inputs['EducationField'] = st.selectbox('EducationField', options=['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other'])
cat_inputs['Gender'] = st.selectbox('Gender', options=['Male', 'Female'])
cat_inputs['JobRole'] = st.selectbox('JobRole', options=['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 
                                                         'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'])
cat_inputs['MaritalStatus'] = st.selectbox('MaritalStatus', options=['Single', 'Married', 'Divorced'])

# Combine inputs into a DataFrame
input_data = {**num_inputs, **cat_inputs}
input_df = pd.DataFrame([input_data])

# Prediction button
if st.button("Predict Attrition"):
    prediction = model.predict(input_df)
    prob = model.predict_proba(input_df)[0][1]  # Probability of attrition (class 1: Yes)
    
    if prediction[0] == 1:
        st.error(f"High Risk of Attrition! Probability: {prob:.2f}")
    else:
        st.success(f"Low Risk of Attrition. Probability: {prob:.2f}")