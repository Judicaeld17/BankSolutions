import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the model
model = joblib.load('model.pkl')

# Title of the app
st.title("Loan Offer Acceptance Predictor")

# File uploader for CSV
uploaded_file = st.file_uploader("Upload CSV file for predictions", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded file into a DataFrame
    data = pd.read_csv(uploaded_file)

    # Display the uploaded data
    st.write("Uploaded Data:")
    st.write(data)

    # Preprocess the data
    data['Securities Account'] = data['Securities Account'].apply(lambda x: 1 if x == "Yes" else 0)
    data['CD Account'] = data['CD Account'].apply(lambda x: 1 if x == "Yes" else 0)
    data['Online'] = data['Online'].apply(lambda x: 1 if x == "Yes" else 0)
    data['CreditCard'] = data['CreditCard'].apply(lambda x: 1 if x == "Yes" else 0)

    # Extract the features from the DataFrame
    features = data[['Age', 'Experience', 'Income', 'ZIP Code', 'Family', 'CCAvg',
                     'Education', 'Mortgage', 'Securities Account', 'CD Account', 
                     'Online', 'CreditCard']]

    # Make predictions for each row in the DataFrame
    predictions = model.predict(features)

    # Display the predictions
    data['Prediction'] = predictions
    st.write("Predictions:")
    st.write(data[['Age', 'Experience', 'Income', 'ZIP Code', 'Family', 'CCAvg',
                   'Education', 'Mortgage', 'Securities Account', 'CD Account',
                   'Online', 'CreditCard', 'Prediction']])
