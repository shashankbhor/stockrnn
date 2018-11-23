# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:25:58 2018

@author: Shashank Bhor
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
import nsepy as nse
from pandas import DataFrame
import talib as ta

# inputs for model
stock_cd='BANKNIFTY'
timesteps=60
pred_size=3
testset_perc=0.05
data_columns=['ADX','RSI','MACD','BETA','Close','Volume','Open']
result_columns=['Open']
ModelfileNm = stock_cd + '_t_' + str(timesteps) +'_' + '_'.join(data_columns) + '.rnnModel'
 

stock_df=nse.get_history(symbol=stock_cd,
                    start=date(2006,1,1), 
                    end=date(2018,11,16), index=True)


headers = stock_df.columns


from sklearn.preprocessing import Imputer
imp=Imputer(strategy='mean')

stock_df=imp.fit_transform(stock_df)
stock_df=pd.DataFrame(stock_df,columns=headers)


close=stock_df.loc[:,'Close'].values
high=stock_df.loc[:,'High'].values
low=stock_df.loc[:,'Low'].values
volume=stock_df.loc[:,'Volume'].values

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


# Creating a data structure with timesteps and 1 output
'''
X_train_ds=[]
y_train_ds=[]


for i in range(timesteps,input_set.shape[0]):
   a=np.append(input_set[i-timesteps:i,0],input_set[i-timesteps:i,1]) 
   X_train_ds.append(a) 
   y_train_ds.append(output_set[i,0]) 
'''

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


# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 75, return_sequences = True, input_shape = (X_train_ds.shape[1], features)))
regressor.add(Dropout(0.2))


# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 75, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 75, return_sequences = True))
regressor.add(Dropout(0.2))


# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 75, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 75))
regressor.add(Dropout(0.2))



# Adding the output layer
regressor.add(Dense(units = pred_size))


#Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mse'])


# Fitting the RNN to the Training set
regressor.fit(X_train_ds, y_train_ds, epochs = 85, batch_size = 30)



# Predictions 
inputs=stock_df_test[data_columns].values
real_stock_price=stock_df_test[result_columns].iloc[timesteps:,:].values



''' need to uncomment for single feature '''
#inputs = inputs.reshape(-1,1)

inputs = sc.transform(inputs)

'''
X_test = []
for i in range(timesteps, inputs.shape[0]):
    b=np.append(inputs[i-timesteps:i, 0],inputs[i-timesteps:i, 1]) 
    X_test.append(b)
'''

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


predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc_y.inverse_transform(predicted_stock_price)

pred_Pred1=predicted_stock_price[:,0]
pred_Mean_Pred2=np.mean(predicted_stock_price,axis=1)
pred_ex_last=predicted_stock_price[-1:,:]
pred_ex_last=pred_ex_last.T

pred_Pred2=np.append(pred_Pred1.tolist(),pred_ex_last.tolist())


# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real '+ stock_cd +' Stock Price')
plt.plot(pred_Pred2, color = 'blue', label = 'Predicted '+stock_cd+' Stock Price')
plt.plot(pred_Mean_Pred2, color = 'm', label = 'Predicted '+stock_cd+' Stock Price')
#plt.plot(predicted_stock_price, color = 'green', label = 'Predicted '+stock_cd+' Stock Price')

plt.title(stock_cd+' Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel(stock_cd+' Stock Price')
plt.legend()
plt.show()

