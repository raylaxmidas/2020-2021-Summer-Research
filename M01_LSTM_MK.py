"""
LSTM Model of pedestrian simulation
This notebook fits an LSTM model to the processed pedestrian data in the processed_data folder The objective is to predict where the pedestrian will end up being in the next frame

The data is [x,y,gate_out,left_neighbours, right_neighbour, angle, speed] for each row of the data
then the data is repeated for lb time intervals, so overal the training data will be a matrix [lb x 7]  

We will predict the three variables [angle, speed, gate_out] at time step t+k, with k>0, so the output is a [3x1] array

Note that: 
lb = look-back window or number of time interval in historical data that we are considering
pred = look-ahead window or number of time interval in the future that we'll look at to make the prediction
angle is the deviation angle from a straight line from the current position to the chosen gate out, -90, 90, please note the sign of this angle
speed is in pixel/frame 

Now, in real-time we won't be able to guest the 'gate_out' of a pedestrian, so we can that this is a latent variable 
we make multiple predictions with different values of 'gate_out', Particle Filter is then applied to tell which would be 
the best guess
"""

#%%

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from keras.models import Sequential, load_model, Model
from keras.layers import LSTM, Dropout, Dense, Input, Concatenate, Conv1D, MaxPooling1D
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
#from data_prep import input_prep, plot_loss

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def plot_loss(history):
    plt.plot(history.history['loss'],label='Training loss')
    plt.plot(history.history['val_loss'],label='Test loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

# %% Function to prepare the data through MinMaxScaling and prepare the look-backs etc.
# here look-backs are from the same pedestrian

def prepare_data(data, ped_list, train_ped, pred_col, split=0.1, pred=10, lb=5):

    #ped_list = list of all pedestrian
    #train_ped = list of pedestrian that we'll use for training 
    #pred_col: the column number for the columns we want to predict
    #split: ratio of testing data (between 0-1)
    #pred: the prediction interval, e.g. pred=1 is predicting for t+1
    #lb: look-back steps: how many time interval are we taking into account when predicting for a pedestrian

    pred_col_index = [data.columns.get_loc(c) for c in pred_col]
    data = data.astype('float32')
    
    #Fit two MixMax Scalers on training data (just fit)
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    data_X = data.iloc[:,2:]
    scaler_X.fit(data_X)

    train_X=[]
    train_y=[]
    test_X = []
    test_y = []
    #Loop through each pedestrian to prepare the data
    for p in ped_list: 
        ped_df = data[data['ID']==p]
        ped_df=ped_df.reset_index(drop=True)
        #Assume the the zeros Angle_dev and Speed to be equal to the previous value
        ped_df['Angle_dev'][ped_df['Angle_dev']==0] = ped_df.iloc[np.where(ped_df['Angle_dev']==0)[0]-1]['Angle_dev'].values
        ped_df['Speed'][ped_df['Speed']==0] = ped_df.iloc[np.where(ped_df['Speed']==0)[0]-1]['Speed'].values
        ped_array = np.array(ped_df)
        #now process the data for each line
        for j in range(len(ped_df)-int(pred+lb-1)):
            ped_x=ped_array[j:lb+j,1:]
            ped_x_transformed = scaler_X.transform(ped_x[:,1:])
            #print(train_dataset)
            ped_y=ped_array[int(pred+lb-1)+j,pred_col_index]
            ped_y_transformed=ped_y
            
            #now store the processed data
            if np.isin(p,train_ped): #if the current pedestrian is in the train_ped list
                #print('in')
                train_X.append(ped_x_transformed)
                train_y.append(ped_y_transformed)
            else: #if not, store to the test_X, and test_y data
                test_X.append(ped_x_transformed)
                test_y.append(ped_y_transformed)

    train_X = np.array(train_X)
    train_y = np.array(train_y)
    #train_y = train_y.reshape(-1,2)
    test_X = np.array(test_X)
    test_y = np.array(test_y)
    #test_y= test_y.reshape(-1,2)
    return (train_X, train_y, test_X, test_y,scaler_X)
#%%
def train_nn(train_X, train_y, test_X, test_y,m,lb):
    #Model creation
    inputt = Input(shape=(train_X.shape[1],train_X.shape[2],), name='t_input')
    #inputs = Input(shape=(train_X2.shape[1],train_X2.shape[2],), name='s_input')
    #c = Concatenate(-1)([inputt,inputs])
    #seg_num = train_X.shape[-1]+train_X2.shape[-1]
    #seg_num = train_X.shape[-1]
    #h1 = Conv1D(filters=64, kernel_size=5, padding='valid', activation='relu', strides=1)(inputt)
    #h2 = MaxPooling1D(pool_size=4)(h1)
    #h1 = Dropout(0.2)(inputt)
    #h2 = Dense(6)(inputt)
    h1 = LSTM(8, return_sequences=False)(inputt)
    h2 = Dense(8)(h1)
    #h3 = LSTM(6, return_sequences=False)(h2)
    #outp1 = Dense(train_X.shape[-1]+1)(h3)
    outp1 = Dense(2)(h2)
    #outp2 = Dense(train_X2.shape[-1])(h3)
    #model = Model(inputs=[inputt,inputs], outputs=[outp1,outp2])
    model = Model(inputs=[inputt], outputs=[outp1])
    #Compile and fit
    model.compile(optimizer='adam', loss='mae', metrics=['mse'])
    history = model.fit({'t_input': np.array(train_X)}, [np.array(train_y)], epochs=1000, batch_size=50, verbose=0, validation_data=([test_X], [test_y]))
    #history = model.fit(train_X, train_y)

    #Plot loss functions
    plot_loss(history)

    figurename = './figures/LSTM_pred' + str(m) + '_lookback' + str(lb) + '.pdf'
    plt.savefig(figurename)

    yhat = model.predict(test_X)
    #yhat_transformed = scaler_Y.inverse_transform(yhat)

    return yhat, model


# %%
import pickle 
#lb = 5 #look-back value

df = pd.read_csv('./processed_data/ped_data_train2.csv')

pred_col=['Angle_dev','Speed']
#pred_col=['Angle_dev']
#pred_col=['Angle_dev']
ped_list = np.unique(df['ID'])
#list of pedestrian for training and testing data
split=0.3
train_ped = np.sort(np.random.choice(ped_list,int((1-split)*len(ped_list)),replace=False))
res = []
for lb in [20]:
    print("Look back = ", lb)
    for m in [1]:

        train_X, train_y, test_X, test_y,scaler_X = prepare_data(df, ped_list, train_ped, pred_col, split, m, lb)
        print("Data proccessed for step ahead = ", m)
        yhat, model= train_nn(train_X, train_y, test_X, test_y,m,lb)
        #calculate the accuracy indexes
        MAE_angle = mean_absolute_error(yhat[:,0], test_y[:,0])
        MAE_speed = mean_absolute_error(yhat[:,1], test_y[:,1])
        #MAE_gate = mean_absolute_error(yhat[:,2], test_y[:,2])
        res.append([m,MAE_angle,MAE_speed])
        #save the figure
        #figurename = './figures/LSTM_pred' + str(m) + '_lookback' + str(lb) + '.pdf'
        #plt.savefig(figurename)
        #save the model
        outfilename = './model/scaler_pred' + str(m) + '_lookback' + str(lb) + '_AngleSpeedv2.pkl'
        pickle.dump( scaler_X, open( outfilename, "wb" ) )

        outfilename = './model/LSTM_pred' + str(m) + '_lookback' + str(lb) + '_AngleSpeedv2.h5'
        model.save(outfilename)

        #scaler_X


        print("save model:" + outfilename)

res = np.array(res)
np.savetxt("./outputs/Performance_AngleSpeed.csv", res,fmt='%10.3f', delimiter=",")

