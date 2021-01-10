import os

thisfile_dir=os.path.dirname(__file__)
TRAIN_DATAPATH=os.path.join(thisfile_dir,'..','inputs','train.csv')
TRAINFOLD_DATAPATH=os.path.join(thisfile_dir,'..','inputs','trainfold.csv')
LABEL='price_range'
MODELS_DIR=os.path.join(thisfile_dir,'..','save_models')