import streamlit as st
import joblib
import pandas as pd

# Load the model
model = joblib.load('model.pkl')

# Center the main title
st.markdown(
    """
    <style>
    .title {
        text-align: center;
    }
    .total-row {
        text-align: center;
        font-size: 20px;  /* Increase font size for visibility */
        font-weight: bold;
    }
    </style>
    <h1 class="title">Prédicteur d'Acceptation d'Offre de Prêt</h1>
    """,
    unsafe_allow_html=True
)

# Create two columns with equal width
col1, col2 = st.columns(2)

with col1:
    # File uploader for CSV
    uploaded_file = st.file_uploader("Téléchargez un fichier CSV pour les prédictions", type=["csv"])

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

    # Format the prediction scores to 2 decimal places
    formatted_scores = [f"{score:.2f}" for score in prediction_scores]

    # Create the DataFrame with prediction scores
    output_data = pd.DataFrame({
        'Âge': data['Age'],
        'Expérience': data['Experience'],
        'Score de Prédiction': formatted_scores
    })

    with col2:
        # Slider to set the threshold
        threshold = st.slider("Définissez le seuil pour la prise de décision", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

        # Update the DataFrame with predictions based on the selected threshold
        output_data['Prédiction'] = ["❌ Non approuvé" if float(score) < threshold else "✅ Approuvé" for score in formatted_scores]

        # Display the updated predictions
        st.write("Prédictions mises à jour avec le seuil :")
        st.write(output_data)

        # Count the number of "Approuvé" and "Non approuvé"
        count_approuve = output_data['Prédiction'].str.contains("✅").sum()
        count_non_approuve = output_data['Prédiction'].str.contains("❌").sum()

        # Create a DataFrame to display the totals
        totals = pd.DataFrame({
            'Prédiction': [
                f"✅ Approuvé: {count_approuve}",
                f"❌ Non approuvé: {count_non_approuve}"
            ]
        })

        # Display the totals in a row at the bottom of the columns
        st.markdown('<div class="total-row">', unsafe_allow_html=True)
        st.write(totals, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    with col2:
        # Show a message if no file has been uploaded
        st.write("Veuillez télécharger un fichier CSV pour commencer les prédictions.")
