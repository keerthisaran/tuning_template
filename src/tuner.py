from skopt import gp_minimize
from skopt import space
import model_dispatcher 
from sklearn import model_selection
from sklearn import metrics
from functools import partial
from sklearn import ensemble

import config
import numpy as np

def optimize(param_values,param_names,df,model_str):

    params=dict(zip(param_names,param_values))
    model=ensemble.RandomForestClassifier(**params)
    # model=model_dispatcher.MODELS_DICT[model_str](**params)
    accuracies=[]
    for i in range(5):
        
        df_train=df.loc[df.kfold!=i]
        X_train=df_train.drop([config.LABEL,'kfold'],axis=1).values
        y_train=df_train[config.LABEL].values
        df_valid=df.loc[df.kfold==i]
        X_valid=df_valid.drop([config.LABEL,'kfold'],axis=1).values
        y_valid=df_valid[config.LABEL]
        print('success')
        model.fit(X_train,y_train)
        print('success2')
        pred=model.predict(X_valid)
        
        fold_accuracy=metrics.accuracy_score(pred,y_valid)
        accuracies.append(fold_accuracy)
    
    return -1*np.mean(accuracies)



def optimizer(df,params,model_str):

    param_names=params.keys()
    param_space=[]
    d={'int':space.Integer,
       'real':space.Real,
       'cat':space.Categorical,
       }
    for param_name in param_names:
        
        t,r=params[param_name]
        param_space.append(d[t](*r))
    print(param_names)
    optimization_func=partial(optimize,param_names=param_names,df=df,model_str=model_str)

    result=gp_minimize(optimization_func,param_space,n_calls=15,n_random_starts=10,verbose=10)
    ret=dict(zip(param_names,result.x))
    return ret

    
