import streamlit as st
import pandas as pd
import numpy as np
import pickle  # Import pickle for loading the model

# Load the saved model
with open('trained_model.sav', 'rb') as model_file:
    model = pickle.load(model_file)

# Define class labels based on your label encoding
class_labels = ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica']

# Define CSS styles for different classes with increased font and box sizes
class_styles = {
    'Iris Setosa': 'background-color: lightgreen; padding: 10px 20px; border: 2px solid green; border-radius: 5px; text-align: center; font-size: 24px; color: darkgreen; display: inline-block; vertical-align: middle;',
    'Iris Versicolor': 'background-color: lightblue; padding: 10px 20px; border: 2px solid blue; border-radius: 5px; text-align: center; font-size: 24px; color: darkblue; display: inline-block; vertical-align: middle;',
    'Iris Virginica': 'background-color: lightcoral; padding: 10px 20px; border: 2px solid red; border-radius: 5px; text-align: center; font-size: 24px; color: darkred; display: inline-block; vertical-align: middle;'
}

# Create a Streamlit web app
st.markdown("<h1 style='text-align: center;'># Iris Classification</h1>", unsafe_allow_html=True)
st.write('This application employs machine learning to predict the Iris flower species from input measurements, serving as a convenient tool for identifying different Iris species.')

# Collect user input
sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.0)
sepal_width = st.slider('Sepal Width (cm)', 2.0, 4.5, 3.0)
petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 4.0)
petal_width = st.slider('Petal Width (cm)', 0.1, 2.5, 1.0)

# Create a feature DataFrame
input_data = pd.DataFrame({
    'sepal_length': [sepal_length],
    'sepal_width': [sepal_width],
    'petal_length': [petal_length],
    'petal_width': [petal_width]
})

# Make predictions using the loaded model
prediction = model.predict(input_data)

# Display the predicted class label based on label encoding, styled based on the class with a larger font size
predicted_class = class_labels[prediction[0]]
styled_class = f"<div style='{class_styles[predicted_class]} font-size: 28px;'>{predicted_class}</div>"

# Display the prediction and the result on the same line
st.markdown(f"<h2 style='text-align: center;'>Predicted Class: {styled_class}</h2>", unsafe_allow_html=True)