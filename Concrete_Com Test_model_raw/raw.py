# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 01:58:35 2023

@author: navee
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model
loaded_model = pickle.load(open('final_model.sav', 'rb'))

# Create a function to make predictions
def make_prediction(cement, age, blast_furnace_slag, coarse_aggregate, fine_aggregate, fly_ash, superplasticizer, water):
    new_data = pd.DataFrame([
        [cement, age, blast_furnace_slag, coarse_aggregate, fine_aggregate, fly_ash, superplasticizer, water]],
        columns=['cement', 'age', 'blast_furnace_slag', 'coarse_aggregate', 'fine_aggregate', 'fly_ash', 'superplasticizer', 'water'])
    prediction = loaded_model.predict(new_data)
    return prediction

# Create the Streamlit app
def main():
    st.title("Concrete Strength Prediction")
    
    st.subheader("Home")
    st.write("Welcome to the Concrete Strength Prediction App. This application predicts the concrete strength based on its constituents.")
    
    st.subheader("Predict the Concrete Strength")
    
    cement = st.number_input("Cement (Kg/m³)", min_value=0.0, max_value=1000.0, step=0.1)
    age = st.slider("Age (1 to 365 days)", min_value=1, max_value=365, step=1)
    blast_furnace_slag = st.number_input("Blast Furnace Slag (Kg/m³)", min_value=0.0, max_value=1000.0, step=0.1)
    coarse_aggregate = st.number_input("Coarse Aggregate (Kg/m³))", min_value=0.0, max_value=1000.0, step=0.1)
    fine_aggregate = st.number_input("Fine Aggregate (Kg/m³))", min_value=0.0, max_value=1000.0, step=0.1)
    fly_ash = st.number_input("Fly Ash (Kg in M^3)", min_value=0.0, max_value=1000.0, step=0.1)
    superplasticizer = st.number_input("Superplasticizer (Kg/m³)", min_value=0.0, max_value=1000.0, step=0.1)
    water = st.number_input("Water (Kg/m³)", min_value=0.0, max_value=1000.0, step=0.1)

    if st.button("Predict"):
        result = make_prediction(cement, age, blast_furnace_slag, coarse_aggregate, fine_aggregate, fly_ash, superplasticizer, water)
        st.write(f"Predicted Concrete Strength: {result[0]:.2f} MPa")
    
    st.subheader("About")
    st.write("This application is built with Streamlit and scikit-learn. The model used is XGBoost Regressor.")

if __name__ == "__main__":
    main()
    
    
    
    
    
    # Create the Streamlit app
def main():
    st.title("Concrete Strength Prediction")
    
    home_expander = st.beta_expander("Home", expanded=True)
    with home_expander:
        st.write("Welcome to the Concrete Strength Prediction App. This application predicts the concrete strength based on its constituents.")
    
    predict_expander = st.beta_expander("Predict")
    with predict_expander:
        st.subheader("Predict the Concrete Strength")
        cement = st.number_input("Cement (Kg/m³)", min_value=0.0, max_value=1000.0, step=0.1)
        age = st.slider("Age (1 to 365 days)", min_value=1, max_value=365, step=1)
        blast_furnace_slag = st.number_input("Blast Furnace Slag (Kg/m³)", min_value=0.0, max_value=1000.0, step=0.1)
        coarse_aggregate = st.number_input("Coarse Aggregate (Kg/m³))", min_value=0.0, max_value=1000.0, step=0.1)
        fine_aggregate = st.number_input("Fine Aggregate (Kg/m³))", min_value=0.0, max_value=1000.0, step=0.1)
        fly_ash = st.number_input("Fly Ash (Kg in M^3)", min_value=0.0, max_value=1000.0, step=0.1)
        superplasticizer = st.number_input("Superplasticizer (Kg/m³)", min_value=0.0, max_value=1000.0, step=0.1)
        water = st.number_input("Water (Kg/m³)", min_value=0.0, max_value=1000.0, step=0.1)

        if st.button("Predict"):
            result = make_prediction(cement, age, blast_furnace_slag, coarse_aggregate, fine_aggregate, fly_ash, superplasticizer, water)
            st.write(f"Predicted Concrete Strength: {result[0]:.2f} MPa")
    
    about_expander = st.beta_expander("About")
    with about_expander:
        st.write("This application is built with Streamlit and scikit-learn. The model used is XGBoost Regressor.")

if __name__ == "__main__":
    main()



# Create the Streamlit app
def main():
    st.title("Concrete Strength Prediction")

    menu = ["Home", "Predict", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        st.write("Welcome to the Concrete Strength Prediction App. Go to the Predict tab to get started.")
        st.subheader("Home")
        st.write("Welcome to the Concrete Strength Prediction App. Go to the Predict tab to get started.")
    
    elif choice == "Predict":
        st.subheader("Predict the Concrete Strength")
        cement = st.number_input("Cement (Kg/m³)", min_value=0.0, max_value=1000.0, value=0.0, step=0.1)
        age = st.slider("Age (1 to 365 days)",min_value=1, max_value=365,step=1)
        blast_furnace_slag = st.number_input("Blast Furnace Slag (Kg/m³)",min_value=0.0, max_value=1000.0, value=0.0, step=0.1)
        coarse_aggregate = st.number_input("Coarse Aggregate (Kg/m³))",min_value=0.0, max_value=1000.0 ,value=0.0, step=0.1)
        fine_aggregate = st.number_input("Fine Aggregate (Kg/m³))",min_value=0.0, max_value=1000.0, value=0.0, step=0.1)
        fly_ash = st.number_input("Fly Ash (Kg in M^3)", min_value=0.0, max_value=1000.0,value=0.0, step=0.1)
        superplasticizer = st.number_input("Superplasticizer (Kg/m³)",min_value=0.0, max_value=1000.0 ,value=0.0, step=0.1)
        water = st.number_input("Water (Kg/m³)", min_value=0.0, max_value=1000.0,value=0.0, step=0.1)

        if st.button("Predict"):
            result = make_prediction(cement, age, blast_furnace_slag, coarse_aggregate, fine_aggregate, fly_ash, superplasticizer, water)
            st.write(f"Predicted Concrete Strength: {result[0]} MPa")
    elif choice == "About":
        st.subheader("About")
        st.write("This app uses a machine learning model trained on concrete data to predict concrete strength.")

if __name__ == "__main__":
    main()