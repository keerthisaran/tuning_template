import pandas as pd
from sklearn import model_selection
import config

def create_folds(folds=5,label='label'):
    train_data_path=config.TRAIN_DATAPATH
    train_df=pd.read_csv(train_data_path)
    train_df=train_df.sample(frac=1).reset_index(drop=True)
    kf=model_selection.StratifiedKFold(n_splits=folds)
    train_df['kfold']=-1
    train_df[label]=train_df[label].astype(str)
    for fold_ind,(train_inds,valid_inds) in enumerate(kf.split(train_df,train_df[label])):
        train_df.loc[valid_inds,'kfold']=fold_ind
    train_df.to_csv(config.TRAINFOLD_DATAPATH,index=False) 


    return train_df

if __name__=='__main__':
    create_folds(5,label='price_range')