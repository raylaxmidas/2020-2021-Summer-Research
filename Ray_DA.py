"""
Summer Research Project 2021 - Data Assimilation
Author: Raynil Laxmidas

This script predicts pedestrain movements based on processed grand central train station
video frames. Pedestrian movements are predicted using a data assimilation technique via 
particle filter which takes in input from a LSTM Model and Real Data. This script requires
the processed train station data (ped_data_all.csv) and the LSTM Model to function.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from math import cos,sin,atan2,degrees,radians, sqrt
from keras.models import load_model
from numpy.random import uniform
from numpy.random import randn
from filterpy.monte_carlo import systematic_resample
from numpy.random import seed

#----------------------------------------------------------------------------------------
#LSTM Functions Below:

"""
This function prepares the processed grand central train station data to be passed into 
the LSTM Model.
Inputs:
df          (data_frame)    = total data from CSV file.
time        (int)           = the time from which data needs to be prepared from.
lb          (int)           = look back window.
scaler      (obj)           = the scaler used against the training data for the LSTM Model
gate_out    (int)           = gate out to impose on the data.

Outputs:
LSTM_Input  (3D Array)      = a 3D array containing scaled data for all the peds at "time"
"""
def prepare_data(df,time,lb,scaler,Gate_out):
    #Filter the data to only include "time - lb" to "time":
    data_filtered = df[(df['Time'] > time-lb) & (df['Time'] <= time)]
    
    #Filter the data to only include the data at current time step:
    data_now = df[df['Time']==time]

    #Find out the pedestrains present at this time step. 
    ped_list_now = pd.unique(data_now['ID'])
    
    #Create a list of pedID used:
    pedsUsed = []
    
    i = 0
    j = 0
    
    #Loop goes through each ped at the indicated time step and scales thems into a 3D Array
    while i < (len(ped_list_now)):                
        #Create a dataset containing only the pedestrain of interest
        ped_df = data_filtered[data_filtered['ID']==ped_list_now[i]]
        ped_df = ped_df.reset_index(drop=True)
    
        #Set Gate Out:
        ped_df["Gate_out"] = Gate_out
    
        #Drop the first column (Time) and second column (ID):
        ped_df = ped_df.drop(ped_df.columns[0],axis=1)
        ped_df = ped_df.drop(ped_df.columns[0],axis=1)
                
        #Transform Data:
        ped_df_scaled = scaler.transform(ped_df)

        #Use len(ped_df_scaled)>4 to ensure peds being used have data for the last five timesteps.         
        if (j == 0) and (len(ped_df_scaled)>4):
            #For first instance create a 2D array of the first ped data in the frame.
            LSTM_Input = ped_df_scaled
            pedsUsed.append(ped_list_now[i])
            j = 1
        elif (j == 1) and (len(ped_df_scaled)>4): 
            #Stack first ped data with the second in the frame forming a 3D array.
            LSTM_Input = np.stack((LSTM_Input,ped_df_scaled))
            pedsUsed.append(ped_list_now[i])
            j = 2
        elif (j == 2) and (len(ped_df_scaled)>4): 
            #Continue appending ped data into 3D array.
            LSTM_Input = np.concatenate((LSTM_Input,[ped_df_scaled]))
            pedsUsed.append(ped_list_now[i])
            
        i += 1
    
    return LSTM_Input, pedsUsed

"""
This function takes the LSTM Prediction of the angle and speed for all peds at a time and 
predicts the other parameters at t+1. The x and y locations are calculated using 
"getNewLocation". The gate in, NL and NR are assumed to be  the same as the previous 
time step. The gate out is forced.

The output is an array with columns: PedID, X, Y, Gate_in, Speed, Angle_dev, Neighbour_left, 
Neighbour_right, and Gate_out.
"""
def predict(LSTMOutput,gateID,time,PedPredXY):
    #Swap Angle and Speed Columns
    LSTMOutput[:, [2, 1]] = LSTMOutput[:, [1, 2]]
    
    #Creat Old X,Y Positions and Gate_Ins Lists
    old_X_Positions = []
    old_Y_Positions = []
    gate_Ins = []
    NLeft = []
    NRight = []
    gate_Outs = []
    i = 0
    
    #Collect Old X,Y Positions, NL, NR and Gate_Ins Lists
    while i < (len(LSTMOutput)):  
        PedData = getPedData(df,LSTMOutput[i][0],time)
             
        if len(PedData) == 1:
            
            #If first prediction is not yet made get old positions from real data:
            if(PedPredXY[0][0] == 0.0 and PedPredXY[0][1] == 0.0):
                old_X_Positions.append(PedData['X'].iloc[0])
                old_Y_Positions.append(PedData['Y'].iloc[0])
            #Otherwise get old position from previously predicted X and Y location:
            else:
                old_X_Positions.append(PedPredXY[0][0])
                old_Y_Positions.append(PedPredXY[0][1])
                
            NLeft.append(PedData['Neighbour_left'].iloc[0])
            NRight.append(PedData['Neighbour_right'].iloc[0])    
            gate_Ins.append(PedData['Gate_in'].iloc[0])
            gate_Outs.append(gateID)
        else:
            old_X_Positions.append(0)
            old_Y_Positions.append(0)    
            NLeft.append(0)
            NRight.append(0)    
            gate_Ins.append(0)
            gate_Outs.append(0)
        
        i = i+1
    
    new_X_Positions = []
    new_Y_Positions = []
    i = 0
    
    #Collect New X,Y Positions
    while i < (len(LSTMOutput)):
        #Store Old X & Y for Ped
        x1 = old_X_Positions[i]  
        y1 = old_Y_Positions[i]
        
        #Get Angle and Speed from LSTM Prediction
        speed = LSTMOutput[i][1]
        angle_dev = LSTMOutput[i][2]
        
        #Calculate X and Y
        x2, y2 = getNewLocation(x1,y1,angle_dev,speed,gateID,gate_locations)
        
        #Store new X and Y Locations:
        new_X_Positions.append(x2)
        new_Y_Positions.append(y2)
        
        i = i+1
    
    #Add columns into the prediction output:    
    LSTMOutput = np.insert(LSTMOutput, 1, new_X_Positions, axis=1)
    LSTMOutput = np.insert(LSTMOutput, 2, new_Y_Positions, axis=1)
    LSTMOutput = np.insert(LSTMOutput, 3, gate_Ins, axis=1)
    LSTMOutput = np.insert(LSTMOutput, 6, NLeft, axis=1)
    LSTMOutput = np.insert(LSTMOutput, 7, NRight, axis=1)
    LSTMOutput = np.insert(LSTMOutput, 8, gate_Outs, axis=1)
    
    t_plus1_Prediction = LSTMOutput
    
    return t_plus1_Prediction

"""
This function takes the prediction: scales it and then adds it to the previously
prepared data. The last last time step is dropped so there are 5 time steps new data. The
output of this function can be passed into the LSTM model again to produce another prediction
of the angle and speed.
"""
def prepareNextLSTMInput(prediction,preparedData):
    #Delete of Index Column of Prediction
    prediction = np.delete(prediction, 0, axis=1)
    
    #Rescale Prediction
    prediction_scaled = scaler.transform(prediction)
    i = 0
    
    #Add Extra Row of Prediction to Prepared Data:
    preparedData = np.pad(preparedData, ((0,0), (0,1), (0, 0)), 'constant')
    
    while i < len(prediction):
        #Slice out prediction for each Ped:
        prediction_slice = prediction_scaled[i:1+i, 0:8]
    
        j = 0
        while j < 8:
            preparedData[i][5][j] = prediction_slice[0][j]
            j = j + 1
        
        i = i + 1
    
    #Remove the last time sequence.
    nextLSTMInput = preparedData[0:len(prediction),1:6,0:8]
    
    return nextLSTMInput

#This function gets the pedestrains present at a given time step:
def getPedIds(df,time,lb):
    #Filter the data to only include the data at current time step:
    data_now = df[df['Time']==time]
    
    #Find out the pedestrains present at this time step. 
    ped_list_now = pd.unique(data_now['ID'])
    return ped_list_now    

#Gets the data for a specific ped at a specific time:
def getPedData (df, pedID, time):
    PedData = df[(df['Time'] == time) & (df['ID'] == pedID)]
    return PedData

#Calculates the next x and y based on the previous x,y,angle and speed.
def getNewLocation (x1,y1,angle_dev,speed,gateID,gate_locations):
    #Get Gate Location:
    gateout_x = gate_locations[gateID][0]
    gateout_y = gate_locations[gateID][1]
    
    #Find Global Angle of current location to Gate Location:
    xDiff = gateout_x - x1
    yDiff = gateout_y - y1
    
    #Find Global Angle:
    globalAngle = degrees(atan2(yDiff, xDiff))
    
    #Calculate Ped Direction
    pedAngle = globalAngle + angle_dev
    
    #Calculate new positions:
    x2 = speed*cos(radians(pedAngle)) + x1
    y2 = speed*sin(radians(pedAngle)) + y1
        
    return x2,y2

'''
Returns the Pedestrian's Predicted X and Y Location in a [x,y] format using the prediction 
array as a input
'''
def newXY(prediction, pedID):
    prediction_df = pd.DataFrame({'ID': prediction[:, 0], 'X': prediction[:, 1], 
                              'Y': prediction[:, 2], 'Gate In': prediction[:, 3], 
                              'Speed': prediction[:, 4], 'Angle_dec': prediction[:, 5], 
                              'NL': prediction[:, 1], 'NR': prediction[:, 6], 
                              'Gate_out': prediction[:, 7]})

    NewLocation = prediction_df[prediction_df['ID']==pedID]
    newXY = np.zeros((1, 2))
    newXY[0,0] = NewLocation.iloc[0]['X']
    newXY[0,1] = NewLocation.iloc[0]['Y'] 
    return newXY

#-----------------------------------------------------------------------------
"""
Particle Filter Functions Adapted from:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
"""

"""
This function creates a set of particles across the uniformly across map.

Particles can have position, heading, and/or whatever other state variable you need to 
estimate. Each has a weight (probability) indicating how likely it matches the actual 
state of the system. Initialize each with the same weight.
"""
def create_uniform_particles(x_range, y_range, hdg_range, N):
    particles = np.empty((N, 3))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = uniform(hdg_range[0], hdg_range[1], size=N)
    particles[:, 2] %= 2 * np.pi
    return particles

#Creates a set of particles across the using a gaussian distribution (function not used): 
def create_gaussian_particles(mean, std, N):
    particles = np.empty((N, 3))
    particles[:, 0] = mean[0] + (randn(N) * std[0])
    particles[:, 1] = mean[1] + (randn(N) * std[1])
    particles[:, 2] = mean[2] + (randn(N) * std[2])
    particles[:, 2] %= 2 * np.pi
    return particles

"""
This function move the particles with some random-ness to have a reasonable chance of 
capturing the actual position of the pedestrain.
"""
def particle_predict(particles, u, std, dt=1.):
    """ move according to control input u (heading change, velocity)
    with noise Q (std heading change, std velocity)`"""

    N = len(particles)
    # update heading
    particles[:, 2] += u[0] + (randn(N) * std[0])
    particles[:, 2] %= 2 * np.pi

    # move in the (noisy) commanded direction
    dist = (u[1] * dt) + (randn(N) * std[1])
    particles[:, 0] += np.cos(particles[:, 2]) * dist
    particles[:, 1] += np.sin(particles[:, 2]) * dist

"""
This function weights those particles closest to the measurement received from the LSTM
or real data.
""" 
def update(particles, weights, landmarks):
    distance = np.sqrt(np.sum(np.square((particles[:, 0:2] - landmarks)), axis=1))
    
    #Normalise distances and weight lower values higher:
    weights = (1/distance)/(sum(1/distance))
    
    return weights


#This function estimate the current state of the system (i.e. the position of the pedestrain).
def estimate(particles, weights):
    """returns mean and variance of the weighted particles"""

    pos = particles[:, 0:2]
    mean = np.average(pos, weights=weights, axis=0)
    var  = np.average((pos - mean)**2, weights=weights, axis=0)
    return mean, var

#A simple resampling methord based on multinomial resampling (this function is not used):
def simple_resample(particles, weights):
    N = len(particles)
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1. # avoid round-off error
    indexes = np.searchsorted(cumulative_sum, randn(N))
    indexes[indexes!=0] -= 1 #avoid index out of bounds error

    # resample according to indexes
    particles[:] = particles[indexes]
    weights.fill(1.0 / N)
    
#Unused function
def neff(weights):
    return 1. / np.sum(np.square(weights))

#Sampling Importance Resampling filter:
def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights.resize(len(particles))
    weights.fill (1.0 / len(weights))

#Function to visualise the particle filter:
def plotPF (InitialParticles, ResampledParticles, p1, p2, xlim, ylim, gateID, pedID, iters, 
			plot_particles, mu, var, savePlot):          
            
    if plot_particles: #Plot Particles
        plt.legend([InitialParticles, ResampledParticles, p1, p2], 
               ['Initial Particles','Resampled Particles','Actual Position', 'Prediction via PF + LSTM + Real Data'], 
               loc=4, numpoints=1)
    else: #Not Plotting Particles
        plt.legend([p1, p2], 
               ['Actual Position', 'Prediction via PF + LSTM'], 
               loc=4, numpoints=1)
      
    #Plotting labels:    
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.xlabel('X Position in Station (Pixels)')
    plt.ylabel('Y Position in Station (Pixels)')
    plt.title('Pedestrian Movement Prediction via DA using PF + LSTM + Real Data')
    plt.suptitle('Number of Iterations: ' + str(iters))
    
	#Printing out plot details:
    print("------------------------")
    print('Gate ID:' + str(gateID))
    print('PedID:' + str(pedID))
    print('Iterations:' +str(iters))
    
	#Save plot to computer:
    if savePlot:
        plt.savefig('Path_GateID_' + str(gateID) + '_PedID_' + str(pedID) 
                    + '_Iters_' + str(iters) + '.jpeg', dpi=250)
    plt.show()

#Function to record the preformance of the model:
def preformance (preformance, time, x, pedID, gateID, iters, mu, PedActXY, var):
          
    #Add row to preformance dataframe:    
    new_row = {'Time':time,
               'PedID':pedID, 
               'GateID':gateID, 
               'Iterations':x, 
               'Final x':mu[0], 
               'Final y':mu[1],
               'Actual x':PedActXY[0],
               'Actual y':PedActXY[1],
               'Variance x':var[0],
               'Variance y':var[1],
               'Final Position x Error':mu[0]-PedActXY[0],
               'Final Position y Error':mu[1]-PedActXY[1],
               'Error_Pixels':sqrt(((mu[0]-PedActXY[0])**2)+((mu[1]-PedActXY[1])**2))
               }
    
    #Append row to the dataframe
    preformance.loc[len(preformance)] = new_row
    
    return preformance

"""
This function runs the DA Model. Most of the lines of code here are used for visualising
the model running and recording the preformance of while the model is running.
"""
def run_DA(PedPredXY,PedActXY,nextLSTMInput,time,pedID,gateID,df,lb,scaler,pedsUsed,preparedData,
            GetReal=25,
            N=100, 
            iters=50, 
            sensor_std_err=0.1,  
            plot_particles=False,
            plot_iteration=False,
            savePlot=False,
            xlim=(0, 750), 
            ylim=(0, 750), 
            initial_x=None, 
            savePre = False):
      
    plt.figure()
    
    #Create Uniforms Particles Across Map:
    particles = create_uniform_particles((0,750), (0,750), (0, 6.28), N)
    
    weights = np.ones(N) / N
    
    #Plotting of Initial Particles:
    if plot_particles:
        alpha = .20
        if N > 5000:
            alpha *= np.sqrt(5000)/np.sqrt(N)           
        InitialParticles = plt.scatter(particles[:, 0], particles[:, 1], alpha=alpha, color='g')
    else:
        InitialParticles = plt.scatter(particles[:, 0], particles[:, 1], alpha=0.0, color='g')
    
    xs = []
    time = time + 1
    
    #Create a data frame to collect information on the preformance of model:             
    column_names = ["Time", 
                    "PedID",
                    "GateID",
                    "Iterations",
                    "Final x",
                    "Final y",
                    "Actual x",
                    "Actual y",
                    "Variance x",
                    "Variance y",
                    "Final Position x Error", 
                    "Final Position y Error",
                    "Error_Pixels"]   
    
    preformanceDF = pd.DataFrame(columns = column_names)
    
    for x in range(iters): 
        
        #Recieving Actual Data:
        if x % GetReal == 0 and x != 0:
            preparedData, pedsUsed = prepare_data(df,time,lb,scaler,gateID)
            LSTMPrediction = LSTM5.predict(preparedData)
            LSTMPredictionIndexed = np.insert(LSTMPrediction, 0, pedsUsed, axis=1)
            
            PedPredXY = np.zeros((1, 2))
            prediction = predict(LSTMPredictionIndexed,gateID,time,PedPredXY)
            nextLSTMInput = prepareNextLSTMInput(prediction,preparedData)
            
            PedPredXY = newXY(prediction, pedID)
            
            PedActXY = df[(df['ID'] == pedID) & (df['Time'] == time)]
            PedActXY = np.array([PedActXY.iloc[0]['X'], PedActXY.iloc[0]['Y']])
            
            particles = create_uniform_particles((PedPredXY[0][0]-200,PedPredXY[0][0]+200), 
                                                 (PedPredXY[0][1]-200,PedPredXY[0][1]+200), 
                                                 (0, 6.28), N)
            
            weights = np.ones(N) / N
            
            if plot_particles:
                alpha = .20
                if N > 5000:
                    alpha *= np.sqrt(5000)/np.sqrt(N)           
                #InitialParticles = plt.scatter(particles[:, 0], particles[:, 1], alpha=alpha, color='g')           
            
        #Move diagonally forward to (x+1, x+1) #(0.2, 1.414)
        particle_predict(particles, u=(10, 1.414), std=(50, 50))      

        #Incorporate LSTM measurements:
        weights = update(particles, weights, landmarks=PedPredXY)
        
        #Resample:
        #if neff(weights) < N/1.5:
        #if x % 1 == 0:
        indexes = systematic_resample(weights)
        resample_from_index(particles, weights, indexes)
        assert np.allclose(weights, 1/N)
        
        mu, var = estimate(particles, weights)
        xs.append(mu)
        
        #Plot the new particles produced:
        if plot_particles:
            ResampledParticles = plt.scatter(particles[:, 0], particles[:, 1], color='k', 
			marker=',', s=1, alpha = 0.075)
        else: 
            ResampledParticles = plt.scatter(particles[:, 0], particles[:, 1], color='k', 
			marker=',', s=1, alpha = 0.0)
            
        #Plot the actual position:    
        p1 = plt.scatter(PedActXY[0], PedActXY[1], marker='^',color='b', s=50, lw=1)
        
        #Plot the predicted positions using PF and LSTM:
        p2 = plt.scatter(mu[0], mu[1], marker='x', color='r')
        
        #Record the preformance:
        preformanceDF = preformance(preformanceDF, time, x, pedID, gateID, iters, mu, PedActXY, var)
        
        #Step Time:
        time = time + 1
            
        #Generate Next LSTM Prediction:
        LSTMPrediction = LSTM5.predict(nextLSTMInput)
        LSTMPredictionIndexed = np.insert(LSTMPrediction, 0, pedsUsed, axis=1)
        
        prediction = predict(LSTMPredictionIndexed,gateID,time,PedPredXY)
        nextLSTMInput = prepareNextLSTMInput(prediction,preparedData)
        
        #Get next pedestrian's predict XY location from the LSTM Model:
        PedPredXY = newXY(prediction, pedID)
        
        #Get next pedestrian's actual location from the original dataframe / CSV file this 
		#if for plotting purposes only:
        PedActXY = df[(df['ID'] == pedID) & (df['Time'] == time)]
        PedActXY = np.array([PedActXY.iloc[0]['X'], PedActXY.iloc[0]['Y']])
        
        #If you want plot each iteration:
        if plot_iteration:
            plotPF (InitialParticles, ResampledParticles, p1, p2, xlim, ylim, gateID, 
                  pedID, x, plot_particles, mu, var, savePlot)
            
            
    xs = np.array(xs)
    
    #Overall plotting of PF and Preformanc:
    plotPF(InitialParticles, ResampledParticles, p1, p2, xlim, ylim, gateID, 
          pedID, iters, plot_particles, mu, var, savePlot)
    
    #Show Preformance:
    preformanceDF.plot.line(x='Iterations', y='Error_Pixels')
    
    #Save preformance:
    if savePre:
        preformanceDF.to_csv('PedID_' + str(pedID) + 'GateID' + str(gateID) 
                         + 'Iter' + str(iters) + '.csv', index = False, header=True)
  
#-----------------------------------------------------------------------------
#Main Script:
#Store the location of the gates:
gate_locations = np.array([[0, 275], [125, 700], [577.5 , 700], [740, 655], [740, 475], 
							[740, 265], [740, 65], [647.5, 0], [462.5, 0], [277.5, 0], 
							[92.5, 0]])

#Load the testing data
df = pd.read_csv('./processed_data/ped_data_all.csv')

#----------------------------------------------------------------------------
#Find the list of agents in the data
agent_list = pd.unique(df['ID'])
pred = 1    #look-ahead window = 1
lb = 5      #look-back window = 5 intervals

#Set start time to analyse from:
time = 3010

#Set GateID and PedID (Example PedID = 7, GateID = 0) to Analyse:
gateID = 0
pedID = 7

#----------------------------------------------------------------------------

#Find the start and end times of the data
time_start = df['Time'][0]+lb
time_end = df['Time'][len(df.index)-1]

#Load the prediction model, where look back = 5
outfilename = './model/LSTM_pred' + str(1) + '_lookback' + str(5) + '_AngleSpeedv2.h5'
LSTM5 = load_model(outfilename)

#Load In Training Data Used for LSTM:
LSTM_training_data = pd.read_csv('./processed_data/ped_data_train2.csv')
    
#Generate scalar based on the training data:
scaler = MinMaxScaler(feature_range=(0, 1))
LSTM_training_data = LSTM_training_data.iloc[:,2:]
scaler.fit(LSTM_training_data)

#-----------------------------------------------------------------------------

#While loop to test all the gates:
while gateID <= 10:

    #1.0 Prepare Data from CSV File
    preparedData, pedsUsed = prepare_data(df,time,lb,scaler,gateID)
    
    #2.0 Make Initial LSTM Prediction for Angle and Speed
    LSTMPrediction = LSTM5.predict(preparedData)
    
    #2.1 Add Index of PedIDs:
    LSTMPredictionIndexed = np.insert(LSTMPrediction, 0, pedsUsed, axis=1)
    
    #3.0 Predict next time step parameters:
    PedPredXY = np.zeros((1, 2)) 
    prediction = predict(LSTMPredictionIndexed,gateID,time,PedPredXY)
    
    #3.1 Pedestrains Pedestrian XY 
    PedPredXY = newXY(prediction, pedID) 
    
    #3.2 Pedestrains Actual XY (Test with PedID = 166)
    PedActXY = df[(df['ID'] == pedID) & (df['Time'] == time)]
    PedActXY = np.array([PedActXY.iloc[0]['X'], PedActXY.iloc[0]['Y']])
    
    #4.0 Prepare the data for the next LSTM input with the new prediction:
    nextLSTMInput = prepareNextLSTMInput(prediction,preparedData)
    
    """
    5.0 Run the DA Model
    GetReal = Number of iterations before receiving real data input, set to a high number 
	to avoid any real data input.
	- N = Number of particles
    - iters = Number of iterations
    - plot_particles = if TRUE plots the particles in visuals.
	- plot_iterations = if TRUE plots each iteration of the particle filter good for GIFs.
	- savePlot = if TRUE will save plots to root folder.
	- savePre = if TRUE will save the preformance to the root folder.
    """
    
    seed(2)
    
    run_DA(PedPredXY,PedActXY,nextLSTMInput,time,pedID,gateID,df,lb,scaler,pedsUsed,preparedData,
            GetReal = 25,
            N=100, 
            iters=350, 
            plot_particles=True,
            plot_iteration=False,
            savePlot=False,
            savePre = False,
            xlim=(0,750), 
            ylim=(0,750),
            )
    
    #Increment Gate ID
    gateID = gateID + 1

