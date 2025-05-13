
# # save models
# with open('E_COMMERCE_frhn/Notebook/scaler.pkl', 'rb') as scaler_file:
#     scaler = pickle.load(scaler_file)
    
# with open('E_COMMERCE_frhn/Notebook/model.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)

# st.title("E-Commerce Sales Predictor")


# import pickle

# with open('../Notebook/scaler.pkl', 'rb') as scaler_file:
#     scaler = pickle.load(scaler_file)

# with open('../Notebook/model.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)

import streamlit as st
import numpy as np
import os
import pickle


base_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the path of app.py

scaler_path = os.path.join(base_dir, '..', 'Notebook', 'scaler.pkl')
model_path = os.path.join(base_dir, '..', 'Notebook', 'model.pkl')

with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)



avg_session_length = st.number_input("Avg. Session Length")
time_on_app = st.number_input("Time on App")
length_of_membership = st.number_input("Length of Membership")


if st.button("Predict"):

    input_data = np.array([avg_session_length, time_on_app , length_of_membership]).reshape(1, -1)
    
    scaled_data = scaler.transform(input_data)
    
    prediction = model.predict(scaled_data)
    
    st.success(f"The predicted value is: {prediction[0]:.2f}")