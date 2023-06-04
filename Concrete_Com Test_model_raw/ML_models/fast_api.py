# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 03:55:48 2023

@author: navee
"""
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json


app = FastAPI()


class model_input(BaseModel):
    
    Cement_Kg_per_m3 : float
    Age_Days : int
    Blast_Furnace_Slag_Kg_per_m3 : float
    Coarse_Aggregate_Kg_per_m3 : float
    Fine_Aggregate_Kg_per_m3 : float
    Fly_Ash_Kg_per_m3 : float
    Superplasticizer_Kg_per_m3 : float
    Water_Kg_per_m3 : float


# load the save model

loaded_model = pickle.load(open(r'D:\INeuron_Projects\Concrete_Com Test Pred\final_model.sav', 'rb'))    


@app.post('/concrete_strength_prediction')
def concrete_pred(input_parameters: model_input):
    

    cement = input_parameters.Cement_Kg_per_m3
    age = input_parameters.Age_Days
    blast_furnace_slag = input_parameters.Blast_Furnace_Slag_Kg_per_m3
    coarse_aggregate = input_parameters.Coarse_Aggregate_Kg_per_m3
    fine_aggregate = input_parameters.Fine_Aggregate_Kg_per_m3
    fly_ash = input_parameters.Fly_Ash_Kg_per_m3
    superplasticizer = input_parameters.Superplasticizer_Kg_per_m3
    water = input_parameters.Water_Kg_per_m3
    
    data = pd.DataFrame(
    [[cement, age, blast_furnace_slag, coarse_aggregate, fine_aggregate, fly_ash, superplasticizer, water]],
    columns=['cement', 'age', 'blast_furnace_slag', 'coarse_aggregate', 'fine_aggregate', 'fly_ash', 'superplasticizer', 'water']
    )
    prediction = loaded_model.predict([data])
    return prediction
    
    
    
    
    
    
    
    
