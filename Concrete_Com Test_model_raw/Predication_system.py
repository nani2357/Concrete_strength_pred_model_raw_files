# -*- coding: utf-8 -*

import numpy as np
import pickle
import pandas as pd

#loading the saved model

load_model = pickle.load(open(r'D:\INeuron_Projects\Concrete_Com Test Pred\final_model.sav', 'rb'))
 
# Create a new DataFrame for your example data
new_data = pd.DataFrame([
    [380, 270, 95, 932, 594, 0, 0, 228],
    [342, 180, 38, 932, 670, 0, 0, 228]],
    columns=['cement', 'age', 'blast_furnace_slag', 'coarse_aggregate', 
             'fine_aggregate', 'fly_ash', 'superplasticizer', 'water'])

# Make a prediction
prediction = load_model.predict(new_data)

# Print the prediction
print(prediction)






