from os import listdir
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import scipy
import numpy as np
from tqdm import tqdm
import random
from process import *

import keras
import tensorflow as tf
from keras.layers import * # Keras is the most friendly Neural Network library, this Kernel use a lot of layers classes
from keras.models import Model
from keras.layers import LSTM,Dropout,Dense,TimeDistributed,Conv1D,MaxPooling1D,Flatten
from keras.models import Sequential
from keras.optimizers import RMSprop
from tqdm import tqdm # Processing time measurement
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split 
from keras import backend as K # The backend give us access to tensorflow operations and allow us to create the Attention class
from keras import optimizers # Allow us to access the Adam class to modify some parameters
from sklearn.model_selection import GridSearchCV, StratifiedKFold # Used to use Kfold to train our model
from keras.callbacks import * # This object helps the model to train in a smarter way, avoiding overfitting
import lightgbm as lgb

import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from IPython.display import Image
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import argparse

parser = argparse.ArgumentParser(description='Input your parameters.')
parser.add_argument('--features', metavar='F', default=['acceleration_x','gyro_x','acceleration_z', 'gyro_z'], type=list,
                   help='List of features to train (Default: all 7 features)')
parser.add_argument('--days', metavar='D', default=[10,20,30], type=list,
                   help='List of days for feature (Default: [10,20,30])')
parser.add_argument('--file', metavar='FILE', default='samples', type=str,
                   help='Path for training data (Default: samples)')

args = parser.parse_args()
FEATURES = args.features
DAYS = args.days
PATH = args.file

def read_predict_file(PATH):
    csvfiles = [f for f in listdir(PATH)]
    samples = list()
    print(csvfiles)
    for csv in csvfiles:
        df = pd.read_csv(PATH+"/"+csv)
        samples.append(df)
    return samples

def get_model(feature):
    pdt_up = lgb.Booster(model_file='Weights/'+ feature + '_model_upper.txt')
    pdt_low = lgb.Booster(model_file='Weights/'+ feature +'_model_lower.txt')
    return pdt_up, pdt_low

def compute_outer(df, feature):
    offset = 0
    df[feature +'_upperband'] = df[feature +'_upperband']*(1-offset)
    df[feature +'_lowerband'] = df[feature +'_lowerband']*(1+offset)
    df['out_of_upper'] = df[feature +'_upperband']-df[feature]
    df['out_of_lower'] = df[feature +'_lowerband']-df[feature]
    df['out_of_upper'] =  df['out_of_upper'].apply(lambda x: max(0,-x))
    df['out_of_lower'] = df['out_of_lower'].apply(lambda x: max(0,x))
    df['total_out'] = df['out_of_upper'] + df['out_of_lower']
    return df

def plot_traces(df,feature):
    trace0 = go.Scatter(
                x = df_id['second'],
                y = df_id[F + '_upperband'],
                mode = 'lines',
                name = 'upper safety band'
            )
    trace1 = go.Scatter(
                x = df_id['second'],
                y = df_id[F + '_lowerband'],
                mode = 'lines',
                name = 'lower safety band'
            )
    trace2 = go.Scatter(
                x = df_id['second'],
                y = df_id[feature],
                mode = 'lines',
                name = 'feature'
            )
    trace3 = go.Scatter(
                x = df_id['second'],
                y = df_id['total_out'],
                mode = 'lines',
                name = 'Trigger'
    )
    return [trace0,trace1,trace2,trace3]

sample_dfs = read_predict_file(PATH)

for df in sample_dfs:
    print('Predicting sample...')
    for index, F in enumerate(FEATURES):
        df_id = generate_predict_data(df,F,DAYS)
        Upperband_model,Lowerband_model = get_model(F)

        ma = df_id.columns.str.contains('MA')
        std = df_id.columns.str.contains('STD')
        features = list(df_id.columns[ma|std].values)

        df_id[F + '_upperband'] = Upperband_model.predict(df_id[features], num_iteration=Upperband_model.best_iteration)
        df_id[F + '_lowerband'] = Lowerband_model.predict(df_id[features], num_iteration=Lowerband_model.best_iteration)
        df_id = compute_outer(df_id.loc[50:,:], F)
        if index ==0:
            df_acc = df_id.copy()
            df_acc['acc'] = df_id['total_out']
        else:
            df_acc['acc'] = df_acc['acc']+df_id['total_out']

        df_acc = normalize_specific(df_acc,'total_out',(0,1))
        #traces = plot_traces(df_id,F)
        layout = dict(title = 'Safety band for ' + F,
              xaxis = dict(title = 'Second'),
              yaxis = dict(title = 'Normalized value'),
              )
        #fig = dict(data = traces, layout=layout)
        #py.plot(fig, filename='line-mode')
        
        #ax = sns.lineplot(x="second", y=F, data=df_id)
        #ax = sns.lineplot(x="second", y=F + '_upperband', data=df_id)#orange left/right
        #ax = sns.lineplot(x="second", y=F + '_lowerband', data=df_id)#purple left/right
        #ax = sns.lineplot(x="second", y='total_out', data=df_id)#purple left/right
        #plt.legend(title='Line', loc='upper left', labels=[F, 'Upper band', 'Lower band', 'Trigger'])
        #plt.ylabel('Normalized_value')
        #plt.show()
    fig, ax = plt.subplots(figsize=(18,12))
    df_acc = normalize_specific(df_acc,'acc',(0,1))    
    #ax = sns.lineplot(x="second", y='Speed', data=df_acc)
    ax = sns.lineplot(x="second", y='acc', data=df_acc)#orange left/right
    plt.xlabel('Second')
    plt.ylabel('Trigger')
    plt.show()

