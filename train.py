from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
from tqdm import tqdm # Processing time measurement
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split 
from keras import backend as K # The backend give us access to tensorflow operations and allow us to create the Attention class
from keras import optimizers # Allow us to access the Adam class to modify some parameters
from sklearn.model_selection import GridSearchCV, StratifiedKFold # Used to use Kfold to train our model
from keras.callbacks import * # This object helps the model to train in a smarter way, avoiding overfitting
import lightgbm as lgb
from process import *

import argparse

parser = argparse.ArgumentParser(description='Input your parameters.')
parser.add_argument('--features', metavar='F', default=['acceleration_x','acceleration_y','acceleration_z','gyro_x','gyro_y','gyro_z','Speed'], type=list,
                   help='List of features to train (Default: all 7 features)')
parser.add_argument('--days', metavar='D', default=[10,20,30], type=list,
                   help='List of days for feature (Default: [10,20,30])')
parser.add_argument('--file', metavar='FILE', default='data/sampled_85.csv', type=str,
                   help='Path for training data (Default: data/sampled_85.csv)')
parser.add_argument('--sample', metavar='SAMPLE', default=0.7, type=float,
                   help='Sample ratio (Default: 0.7)')


args = parser.parse_args()
FEATURES = args.features
DAYS = args.days
PATH = args.file
DROP_PERC = args.sample

def save_model(trained_model,feature):
    for model in trained_model:
        model.save_model(feature + '_upper2.txt')
        model.save_model(feature + '_lower2.txt')

num_round = 50000
max_depth = 12
num_leaves = 12
num_threads = 8
bagging_freq = 5
bagging_fraction = 0.8
param = {
        #'learning_rate':0.01,
        'bagging_freq': bagging_freq,
        'bagging_fraction': bagging_fraction,
        'boost_from_average':'false',
        'boosting': 'gbdt',
        'feature_fraction': 0.8,
        'learning_rate': 0.0001,
        'max_depth': max_depth,  
        'metric':'mse',
        'min_data_in_leaf': 10,
        'min_sum_hessian_in_leaf': 10.0,
        'num_leaves': num_leaves,
        'num_threads': num_threads,
        'tree_learner': 'serial',
        'objective': 'regression', 
        'verbosity': 1,

    }

model_to_save = list()

for feature in FEATURES:
    train_df, val_df, test_df = get_data(PATH,feature,DAYS,DROP_PERC)
    ma = train_df.columns.str.contains('MA')
    std = train_df.columns.str.contains('STD')
    features = list(train_df.columns[ma|std].values)

    print(features)
    target_upper = train_df[feature + '_Upperlabel']
    val_target_upper = val_df[feature + '_Upperlabel']
    target_lower = train_df[feature + '_Lowerlabel']
    val_target_lower = val_df[feature + '_Lowerlabel']

    print(train_df[features])
    print(target_upper)

    trn_data_upper = lgb.Dataset(train_df[features], label=target_upper)
    val_data_upper = lgb.Dataset(val_df[features], label=val_target_upper)
    model_upper = lgb.train(param, trn_data_upper, num_round, valid_sets=[trn_data_upper, val_data_upper],
        verbose_eval=5000, early_stopping_rounds=10000)
    model_to_save.append(model_upper)

    trn_data_lower = lgb.Dataset(train_df[features], label=target_lower)
    val_data_lower = lgb.Dataset(val_df[features], label=val_target_lower)
    model_lower = lgb.train(param, trn_data_lower, num_round, valid_sets=[trn_data_lower, val_data_lower],
        verbose_eval=5000, early_stopping_rounds=10000)
    model_to_save.append(model_lower)

    print('Saving model...')
    save_model(model_to_save,feature)
