import streamlit as st
import joblib
import pandas as pd

# Load the model
model = joblib.load('model.pkl')

# Title of the app
st.title("Loan Offer Acceptance Predictor")

# Create two columns with equal width
col1, col2 = st.columns(2)

with col1:
    # File uploader for CSV
    uploaded_file = st.file_uploader("Upload CSV file for predictions", type=["csv"])

with col2:
    # Placeholder for content to be updated
    placeholder = st.empty()

if uploaded_file is not None:
    # Read the uploaded file into a DataFrame
    data = pd.read_csv(uploaded_file)

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

    # Create the DataFrame with prediction scores
    output_data = pd.DataFrame({
        'ID': data.index,  # Assuming the ID is the index or you have an 'ID' column
        'Age': data['Age'],
        'Experience': data['Experience'],
        'Prediction Score': prediction_scores
    })

    with col2:
        # Display the uploaded data
        st.write("Uploaded Data:")
        st.write(data)

        # Slider to set the threshold
        threshold = st.slider("Set the threshold for decision-making", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

        # Update the DataFrame with predictions based on the selected threshold
        output_data['Prediction'] = ["❌ Non approuvé" if score < threshold else "✅ Approuvé" for score in prediction_scores]

        # Display the updated predictions
        st.write("Updated Predictions with Threshold:")
        st.write(output_data)

        # Count the number of "Approuvé" and "Non approuvé"
        count_approuve = output_data['Prediction'].str.contains("✅").sum()
        count_non_approuve = output_data['Prediction'].str.contains("❌").sum()

        # Display the counts
        st.write(f"Total '✅ Approuvé': {count_approuve}")
        st.write(f"Total '❌ Non approuvé': {count_non_approuve}")

else:
    with col2:
        # Show a message if no file has been uploaded
        st.write("Please upload a CSV file to start the predictions.")
