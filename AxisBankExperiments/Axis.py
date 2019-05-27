# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 12:47:12 2018

@author: Shashank
For AXISBANK using Banknifty 

Layers - 4 + 1 
Indicators :- 'Volume','BANKNIFTY','CANDLE_HEIGHT','REDORGREEN','Open'
Optimizer :- adadelta
Batch :- 40 epoch = 70 Neurons - 75

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
import nsepy as nse
from pandas import DataFrame
import talib as ta


def buildModel(input_shape,pred_size=1,optimizer='adam'):
    regressor = Sequential()
    regressor.add(LSTM(units = 75, return_sequences = True, input_shape = input_shape))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 75, return_sequences = True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 75))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units = pred_size))
    regressor.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics=['mse'])
    return regressor

def buildTAindicators(stock_df):
    stock_df['REDORGREEN']=stock_df.loc[:,'Open'].subtract(stock_df.loc[:,'Close']).apply(lambda x: 1 if x > 0 else 0).values
    stock_df['CANDLE_HEIGHT']=stock_df.loc[:,'Open'].subtract(stock_df.loc[:,'Close']).values
    stock_df['NATR']= ta.NATR(stock_df.loc[:,'High'].values,stock_df.loc[:,'Low'].values,stock_df.loc[:,'Close'].values,5)
    close=stock_df.loc[:,'Close'].values
    high=stock_df.loc[:,'High'].values
    low=stock_df.loc[:,'Low'].values
    volume=pd.Series.astype(stock_df.loc[:,'Volume'], dtype='double').values
    adx=ta.ADX(high,low,close,timeperiod=14)
    macd, macdsignal, macdhist = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    mfi = ta.MFI(high, low, close, volume, timeperiod=14)
    rsi= ta.RSI(close, timeperiod=14)
    beta = ta.BETA(high, low, timeperiod=5)
    stock_df['ADX']=pd.Series(adx).values
    stock_df['MFI']=pd.Series(mfi).values
    stock_df['RSI']=pd.Series(rsi).values
    stock_df['MACD']=pd.Series(macd).values
    stock_df['BETA']=pd.Series(beta).values
    stock_df=stock_df.iloc[33:,:]
    return stock_df


def initiatizeDataSet(stock_cd,timesteps,pred_size=1,testset_percent=0.01):
    stock_cd=stock_cd
    timesteps=timesteps
    pred_size=pred_size
    testset_percent=testset_percent
    return stock_cd,timesteps,pred_size,testset_percent


# inputs for model
'''stock_cd='AXISBANK'
timesteps=60
pred_size=10
testset_perc=0.01'''
stock_cd,timesteps,pred_size,testset_perc=initiatizeDataSet('AXISBANK',60,10,0.01)

#data_columns=['ADX','RSI','MACD','BETA','REDORGREEN','Close','Volume','Open']
data_columns=['Volume','CANDLE_HEIGHT','REDORGREEN','Open']
result_columns=['Open']

ModelfileNm = stock_cd + '_t_' + str(timesteps) +'_' + '_'.join(data_columns) + '.rnnModel'


stock_df=nse.get_history(symbol=stock_cd,
                    start=date(2010,1,1), 
                    end=date(2018,12,5))
stock_df=stock_df.loc[:,['Open','Close','High','Low','Volume']]

stock_df1=nse.get_history(symbol=stock_cd,
                    start=date(2018,12,6), 
                    end=date(2018,12,18))
stock_df1=stock_df1.loc[:,['Open']]

stock_df=buildTAindicators(stock_df)


headers = stock_df.columns
from sklearn.impute import SimpleImputer
imp=SimpleImputer(strategy='mean')
stock_df=imp.fit_transform(stock_df)
stock_df=pd.DataFrame(stock_df,columns=headers)



# Dividing the training and testing data
df_len=stock_df.shape[0]
df_head=int(df_len*(1-testset_perc))
df_tail=df_len-df_head+timesteps
stock_df_train=stock_df.head(df_head+pred_size)
stock_df_test=stock_df.tail(df_tail)


# Get the usable columns data
input_set=stock_df_train[data_columns].values
output_set=stock_df_train[result_columns].values
features=len(data_columns)


# Feature scaling 
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(-1, 1))
input_set=sc.fit_transform(input_set)

sc_y=MinMaxScaler(feature_range=(-1, 1))
output_set=sc_y.fit_transform(output_set)

X_train_ds=[]
y_train_ds=[]
for i in range(timesteps,input_set.shape[0]-pred_size):
    k=0
    tempX,tempY=[],[]
    while(k<features):
        tempX.extend(input_set[i-timesteps:i,k])
        k+=1
    tempY.extend(output_set[i:i+pred_size,0])
    tempX=np.array(tempX)
    tempY=np.array(tempY)
    X_train_ds.append(tempX)
    y_train_ds.append(tempY)

    

X_train_ds, y_train_ds = np.array(X_train_ds), np.array(y_train_ds)

# Reshaping
X_train_ds = np.reshape(X_train_ds,(X_train_ds.shape[0],int(X_train_ds.shape[1]/features),features))


#  Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.optimizers import Adadelta


from time import time
from keras.callbacks import TensorBoard

# Adding tensorboard
#tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
modelDisct={}

# Model -1 : Fix learning rate 
tensorboard1 = TensorBoard(log_dir="logs/{}".format(time()))
optimizers=SGD(lr=0.1, momentum=0.0, decay=0.0, nesterov=False)
fixLearningRateModel=buildModel(input_shape=(X_train_ds.shape[1], features),pred_size=10,optimizer=optimizers)
fixLearningRateModel.fit(X_train_ds, y_train_ds, epochs = 70, batch_size = 40,  verbose=1, callbacks=[tensorboard1])
modelDisct['FixLearningRateSGD']=fixLearningRateModel

# Model -1 : Adaptive Learing rate - Adam 
tensorboard2 = TensorBoard(log_dir="logs/{}".format(time()))
adamLearningRateModel=buildModel(input_shape=(X_train_ds.shape[1], features),pred_size=10)
adamLearningRateModel.fit(X_train_ds, y_train_ds, epochs = 70, batch_size = 40,  verbose=1, callbacks=[tensorboard2])
modelDisct['adam']=adamLearningRateModel


# Model -2 : Adadelta
tensorboard3 = TensorBoard(log_dir="logs/{}".format(time()))
adadeltaLearningRateModel=buildModel(input_shape=(X_train_ds.shape[1], features),pred_size=10,optimizer='adadelta')
adadeltaLearningRateModel.fit(X_train_ds, y_train_ds, epochs = 70, batch_size = 40,  verbose=1, callbacks=[tensorboard3])
modelDisct['adadelta']=adadeltaLearningRateModel


# Model -2 : Adagrad
tensorboard4 = TensorBoard(log_dir="logs/{}".format(time()))
adagradLearningRateModel=buildModel(input_shape=(X_train_ds.shape[1], features),pred_size=10,optimizer='adagrad')
adagradLearningRateModel.fit(X_train_ds, y_train_ds, epochs = 70, batch_size = 40,  verbose=1, callbacks=[tensorboard4])
modelDisct['adagrad']=adagradLearningRateModel

# Model -2 : RMSprop
tensorboard5 = TensorBoard(log_dir="logs/{}".format(time()))
rmspropLearningRateModel=buildModel(input_shape=(X_train_ds.shape[1], features),pred_size=10,optimizer='rmsprop')
rmspropLearningRateModel.fit(X_train_ds, y_train_ds, epochs = 70, batch_size = 40,  verbose=1, callbacks=[tensorboard5])
modelDisct['rmsprop']=rmspropLearningRateModel


# Model Storage 
from sklearn.externals import joblib
#joblib.dump(regressor, ModelfileNm) 
#load_model = joblib.load(ModelfileNm)

# Predictions 
inputs=stock_df_test[data_columns].values
real_stock_price=stock_df_test[result_columns].iloc[timesteps:,:].values



''' need to uncomment for single feature '''
#inputs = inputs.reshape(-1,1)

inputs = sc.transform(inputs)

X_test = []
for i in range(timesteps,inputs.shape[0]):
    k=0
    temp=[]
    while(k<features):
        temp.extend(inputs[i-timesteps:i,k])
        k+=1
    temp=np.array(temp)
    X_test.append(temp)
    

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], int(X_test.shape[1]/features), features))

modelResults={}
for model in modelDisct.keys():
    modelResults[model]=sc_y.inverse_transform(modelDisct.get(model).predict(X_test))

predicted_stock_price = rmspropLearningRateModel.predict(X_test)
predicted_stock_price = sc_y.inverse_transform(predicted_stock_price)

from scipy.ndimage.filters import gaussian_filter1d
for model in modelDisct.keys():
    predicted_stock_price=modelResults.get(model)
    price_tail=predicted_stock_price[predicted_stock_price.shape[0]-1,1:predicted_stock_price.shape[1]]
    r1=np.append(real_stock_price,price_tail)
    first_day_pred=predicted_stock_price[:,1]
    r2=np.append(first_day_pred,price_tail)
    # Visualising the results
    #plt.plot(gaussian_filter1d(r1, sigma=2), color = 'red',label = 'Real '+ stock_cd +' Stock Price')
    plt.plot(r1, color = 'red',label = 'Real '+stock_cd+' Stock Price')
    plt.plot(r2, color = 'blue',label = 'Predicted '+stock_cd+' Stock Price')
    #plt.plot(predicted_stock_price, color = 'green', label = 'Predicted '+stock_cd+' Stock Price')
    plt.title(stock_cd+' Stock Price Prediction with ' + model)
    plt.xlabel('Time')
    plt.ylabel(stock_cd+' Stock Price')
    plt.legend()
    plt.show()


    

