import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the model
model = joblib.load('model.pkl')

# Title of the app
st.title("Loan Offer Acceptance Predictor")

# Slider to set the threshold
threshold = st.slider("Set the threshold for decision-making", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

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
    prediction_probs = model.predict_proba(features)

    # Extract the probability of the positive class (Class 1)
    prediction_scores = prediction_probs[:, 1]

    # Map predictions to "Non approuvé" and "Approuvé" based on the threshold
    prediction_labels = ["Non approuvé" if score < threshold else "Approuvé" for score in prediction_scores]

    # Create the output DataFrame
    output_data = pd.DataFrame({
        'ID': data.index,  # Assuming the ID is the index or you have an 'ID' column
        'Age': data['Age'],
        'Experience': data['Experience'],
        'Prediction Score': prediction_scores,
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
