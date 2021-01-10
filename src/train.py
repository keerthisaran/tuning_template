import pandas as pd
import os
import config
import model_dispatcher
from tuner import optimize
from functools import partial
from tuner import optimizer
import joblib

import collections
def train(model_str):
    
    train_df=pd.read_csv(config.TRAINFOLD_DATAPATH)

    params={
        'max_depth':['int',[3,15]],
        'n_estimators':['int',[100,1500]],
        'criterion':['cat',[['gini','entropy']]],
        'max_features':['real',[0.01,1]]
            }

    result=optimizer(df=train_df,params=params,model_str=model_str)
    
    joblib.dump(result,os.path.join(config.MODELS_DIR,f'rf_best_model.joblib'))

    # best_parametr

if __name__ == "__main__":
    train_df=pd.read_csv(config.TRAINFOLD_DATAPATH)

    train('rf')
    

