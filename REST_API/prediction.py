import pickle
import pandas as pd
import json

def predict_mpg(config):
    ##loading the model from the saved file
    pkl_filename = "model.pkl"
    with open(pkl_filename, 'rb') as f_in:
        model = pickle.load(f_in)

    if type(config) == dict:
        df = pd.DataFrame(config)
    else:
        df = config
    
    y_pred = model.predict(df)
    
    if y_pred == 0:
        return 'This is a Hateful post'
    elif y_pred == 1:
        return 'This is not a Hateful text'   
    
