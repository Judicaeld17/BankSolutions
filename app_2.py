if uploaded_file is not None:
    # Read the uploaded file into a DataFrame
    data = pd.read_csv(uploaded_file)

    # Display the uploaded data
    st.write("Uploaded Data:")
    st.write(data)

    # Debug: List all columns
    st.write("Columns in uploaded file:", data.columns.tolist())

    # Define required columns
    required_columns = ['Age', 'Experience', 'Income', 'ZIP_Code', 'Family', 'CCAvg',
                        'Education', 'Mortgage', 'Securities Account', 'CD Account', 
                        'Online', 'CreditCard']

    # Check for missing columns
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        st.error(f"The following required columns are missing from the uploaded file: {missing_columns}")
    else:
        # Preprocess the data
        data['Securities Account'] = data['Securities Account'].apply(lambda x: 1 if x == "Yes" else 0)
        data['CD Account'] = data['CD Account'].apply(lambda x: 1 if x == "Yes" else 0)
        data['Online'] = data['Online'].apply(lambda x: 1 if x == "Yes" else 0)
        data['CreditCard'] = data['CreditCard'].apply(lambda x: 1 if x == "Yes" else 0)

        # Extract the features from the DataFrame
        features = data[required_columns]

        # Make predictions for each row in the DataFrame
        predictions = model.predict(features)

        # Display the predictions
        data['Prediction'] = predictions
        st.write("Predictions:")
        st.write(data)
