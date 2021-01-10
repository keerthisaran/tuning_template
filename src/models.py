from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
import joblib

class Model(ABC):

    @abstractmethod
    def fit(self,kwargs):
        pass

    @abstractmethod
    def save(self,path):
        pass

    @abstractmethod
    def predict(self,path):
        pass

class RF(Model):

    def __init__(self,*args,**kwargs):
        self.rf=RandomForestClassifier(*args,**kwargs)

    def fit(self,X,y):
        self.rf.fit(X,y)
        return self.rf
    
    def save(self,path):
        joblib.dump(self.rf,path)

    def predict(self,X):
        return self.rf.predict(X)
    
    def get_params(self,*args,**kwargs):
        return self.rf.get_params(*args,**kwargs)

    def set_params(self,*args,**kwargs):
        return self.rf.set_params(*args,**kwargs)

if __name__=='__main__':
    import pandas as pd
    import config
    df=pd.read_csv(config.TRAINFOLD_DATAPATH)
    X_df=df.drop('price_range',axis=1)
    y=df.price_range
    rf=RF(10)
    rf.fit(X_df,y)
    rf.predict(X_df)
    
    





