from os import listdir
from os.path import isfile, join
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import scipy
import numpy as np
from tqdm import tqdm
import random
#csvfiles = [f for f in listdir("features")]
#print(csvfiles)

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
print(keras.__version__)
print(tf.__version__)
import lightgbm as lgb

BAND_GAP_RATIO = 0.3
SPLIT_RATIO = 0.2

def random_sample(df, labels, drop):

	# Randomly sample the row to drop
	dropID = random.sample(list(labels.index),int(len(labels)*drop))
	dropID = map(float, dropID)

	#Sampling
	df_sampled = df.set_index('bookingID').drop(dropID)
	print('shape after sample:' + str(df_sampled.shape))
	return df_sampled

def findLabel(x,label_dict):
    try:
        return label_dict[x]
    except KeyError:
        return np.nan

def read_data(path,drop):
	filenames = listdir(path)
	csvs = [ filename for filename in filenames if filename.endswith('.csv') ]
	df_all = pd.read_csv(path + "/"+csvs[0])
	for csv in csvs[1:]:
	    print("getting "+ csv + "...")
	    df = pd.read_csv(path+"/"+csv)
	    df_all = df_all.append(df, ignore_index=True)

	#Merge, sort and clean the data
	df_drives = df_all.sort_values(['bookingID', 'second'], ascending=[True, True])
	label_dict, labels = read_label()
	df_drives['label'] = df_drives['bookingID'].apply(lambda x: findLabel(x,label_dict))
	df_drives = df_drives.reset_index(drop=True)
	df_drives = df_drives.loc[:, ~df_drives.columns.str.contains('^Unnamed')]
	df_drives = df_drives.dropna()
	df_sampled = random_sample(df_drives, labels, drop)
	return df_drives.set_index('bookingID')

def read_label():
	label = pd.read_csv("data/part-00000-e9445087-aa0a-433b-a7f6-7f4c19d78ad6-c000.csv")
	label_clear = label.sort_values("bookingID",ascending=True).drop_duplicates('bookingID',keep=False).reset_index(drop=True).set_index("bookingID")
	label_clear['label'] = label_clear['label'].apply(lambda x:int(x))
	label_dict = label_clear.to_dict()
	label_dict = label_dict['label']
	return label_dict, label_clear

def get_moving_average(df,day):
	return df.rolling(window=int(day)).mean()

def get_standard_deviation(df,day):
	return df.rolling(window=int(day)).std()

def get_label(df,ratio):
	uplabel = df + df*BAND_GAP_RATIO
	lowlabel = df - df*BAND_GAP_RATIO
	return uplabel, lowlabel

def arrange(df):
	df.loc[-1]=[0 for i in range(len(df.columns))]
	df.index = df.index+1
	df = df.sort_index()
	df = df.dropna().reset_index(drop=True)
	return df

def train_val_split(df,ratio):
	#df=df.drop('bookingID',axis=1)
	df.dropna(inplace=True)
	train_df = df[:int(round(-(ratio)*df.shape[0]))]
	val_df = df[int(round(-(ratio)*df.shape[0])):]
	return train_df, val_df


def normalize(df , feature):
	normalizer = MinMaxScaler(feature_range=(0.2,0.7))
	df[feature]=normalizer.fit_transform(df[feature].values.reshape(-1, 1))
	return df

def normalize_specific(df , feature, num):
	normalizer = MinMaxScaler(feature_range=num)
	df[feature]=normalizer.fit_transform(df[feature].values.reshape(-1, 1))
	return df

def get_sampled_id(df):
	return list(df.index.drop_duplicates(keep='first'))

def generate_data(df,feature,days):
	dtrain = pd.DataFrame([[0 for i in range(len(df.columns))]])
	dtest = pd.DataFrame([[0 for i in range(len(df.columns))]])
	dtrain.columns = df.columns
	dtest.columns = df.columns
	label_dict, _ = read_label()
	sampleID = get_sampled_id(df)

	for ID in tqdm(sampleID[:100]):
		df_id = df[df.index==ID].reset_index(drop=True)

		if(df_id.shape[0] > 0):
			df_id = normalize(df_id,feature)

			for day in days:
				df_id[feature+'_'+str(day)+'MA'] = get_moving_average(df_id[feature],day)
				df_id[feature+'_'+str(day)+'STD'] = get_standard_deviation(df_id[feature],day)
				df_id[feature+'_Upperlabel'],df_id[feature+'_Lowerlabel'] = get_label(df_id[feature],BAND_GAP_RATIO)

			df_id = arrange(df_id)
			try:
				if label_dict[ID] == 0:
					dtrain = dtrain.append(df_id, ignore_index=True)
				else:
					dtest = dtest.append(df_id, ignore_index=True)
			except KeyError:
				continue
	return dtrain,dtest

def generate_predict_data(df,feature,days):
	df_id = normalize(df,feature)

	for day in days:
		df_id[feature+'_'+str(day)+'MA'] = get_moving_average(df_id[feature],day)
		df_id[feature+'_'+str(day)+'STD'] = get_standard_deviation(df_id[feature],day)

	df_id = arrange(df_id)

	return df_id

def get_data(path,feature,days,drop):
	sampled_df = read_data(path,drop)
	dtrain,dtest = generate_data(sampled_df,feature,days)
	train_df, val_df = train_val_split(dtrain, SPLIT_RATIO)
	return train_df, val_df, dtest
