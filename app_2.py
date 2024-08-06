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

    # Map predictions to "Non approuvé" and "Approuvé"
    prediction_labels = ["Non approuvé" if pred == 0 else "Approuvé" for pred in predictions]

    # Create the output DataFrame
    output_data = pd.DataFrame({
        'ID': data.index,  # Assuming the ID is the index or you have an 'ID' column
        'Age': data['Age'],
        'Experience': data['Experience'],
        'Prediction Score': predictions,
        'Prediction': prediction_labels
    })

    # Display the predictions
    st.write("Predictions:")
    st.write(output_data)

    # Count the number of "Approuvé" and "Non approuvé"
    count_approuve = prediction_labels.count("Approuvé")
    count_non_approuve = prediction_labels.count("Non approuvé")

    # Display the counts
    st.write(f"Total 'Approuvé': {count_approuve}")
    st.write(f"Total 'Non approuvé': {count_non_approuve}")
