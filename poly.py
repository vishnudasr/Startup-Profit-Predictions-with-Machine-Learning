import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Load the dataset

def load_data():
    try:
        dataset = pd.read_csv('salary.csv')
    except FileNotFoundError:
        st.error("Dataset not found. Please make sure 'salary.csv' exists.")
        return None
    return dataset

# Main function
def main():
    st.title("Profit Prediction for Startups")

    # Load the dataset
    dataset = load_data()
    if dataset is None:
        return

    # Display the dataset
    st.subheader("Startup Dataset")
    st.write(dataset)

    # Separate features and target variable
    X = dataset.iloc[:, :-1]  # Features: Position, Level
    y = dataset.iloc[:, -1]    # Target variable: Salary

    # Convert categorical variable 'Position' to numeric using LabelEncoder
    label_encoder = LabelEncoder()
    X['Position'] = label_encoder.fit_transform(X['Position'])

    # Encoding categorical variable 'Position'
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
    X_encoded = ct.fit_transform(X)

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.5, random_state=0)

    # Training the Multiple Linear Regression model on the Training set
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Checking the accuracy on the Test set
    accuracy = regressor.score(X_test, y_test)

    st.subheader("Model Evaluation")
    st.write("Accuracy:", accuracy)

    # Option to make predictions
    st.subheader("Make Predictions")
    st.write("Enter startup details to predict the profit:")

    position = st.selectbox("Position", dataset['Position'].unique())
    level = st.number_input("Level")

    # Convert the selected position to its corresponding label
    position_label = label_encoder.transform([position])[0]

    # Create a feature vector for prediction
    startup_details = np.array([[position_label, level]])

    # Encode the selected position
    startup_details_encoded = ct.transform(startup_details)

    # Make prediction
    if st.button("Predict"):
        prediction = regressor.predict(startup_details_encoded)
        st.write("Predicted Salary:", prediction[0])

if __name__ == "__main__":
    main()
