import pandas as pd
import os
import config
import model_dispatcher
from sklearn import model_selection


def train(model_str):
    
    train_df=pd.read_csv(config.TRAIN_DATAPATH)
    X=train_df.drop(config.LABEL,axis=1)
    y=train_df[config.LABEL]

    model=model_dispatcher.MODELS_DICT[model_str](n_jobs=-1)
    param_grid={
        'n_estimators':[100,200,250,300,400,500],
        'max_depth':[1,2,5,7,11,15],
        'criterion': ['gini','entropy']
    }

    model_cv=model_selection.GridSearchCV(estimator=model,
                                         param_grid=param_grid,
                                         scoring='accuracy',
                                         verbose=10,
                                         n_jobs=1,
                                         cv=5)

    model_cv.fit(X,y)
    
    print(f'best models accuracy is {model_cv.best_score_}')
    # best_parametr

if __name__ == "__main__":
    train('rf')
    

