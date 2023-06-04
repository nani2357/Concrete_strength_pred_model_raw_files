# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 23:23:06 2023

@author: navee
"""

import numpy as np
import pickle
import streamlit as st
import pandas as pd

#loading the saved model

loaded_model = pickle.load(open(r'D:\INeuron_Projects\Concrete_Com Test Pred\final_model.sav', 'rb'))

# creating a function

def make_prediction(cement, age, blast_furnace_slag, coarse_aggregate, fine_aggregate, fly_ash, superplasticizer, water):
    new_data = pd.DataFrame([
        [cement, age, blast_furnace_slag, coarse_aggregate, fine_aggregate, fly_ash, superplasticizer, water]],
        columns=['cement', 'age', 'blast_furnace_slag', 'coarse_aggregate', 'fine_aggregate', 'fly_ash', 'superplasticizer', 'water'])
    prediction = loaded_model.predict(new_data)
    return prediction

# Create the Streamlit app
def main():
    st.title("Concrete Strength Prediction")
    cement = st.number_input("Cement")
    age = st.number_input("Age")
    blast_furnace_slag = st.number_input("Blast Furnace Slag")
    coarse_aggregate = st.number_input("Coarse Aggregate")
    fine_aggregate = st.number_input("Fine Aggregate")
    fly_ash = st.number_input("Fly Ash")
    superplasticizer = st.number_input("Superplasticizer")
    water = st.number_input("Water")

    if st.button("Predict"):
        result = make_prediction(cement, age, blast_furnace_slag, coarse_aggregate, fine_aggregate, fly_ash, superplasticizer, water)
        st.write(f"Predicted Concrete Strength: {result[0]}")

if __name__ == "__main__":
    main()
    