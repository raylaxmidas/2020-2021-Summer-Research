# 2020-2021-Summer-Research
Dynamic Agent-Based Modelling with Data Assimilation in Urban Areas

The objective of this research was to predict pedestrian movements in grand central train station through:

- Creating a **Long short-term memory (LSTM)** artificial recurrent neural network (RNN) that predicts pedestrian angle and movement speed based on 5 previous frames of video data.
- Use a **particle filter (PF)** to incorporate LSTM inputs and real-time data to make predictions of pedestrian movements and the gate at which they will exit the train station.
- Compare the performance of the **data assimilation (DA)** model using PF + LSTM to using a LSTM only model and a random walk model.

Typically, prediction models are trained based on previous historical data and then used to make predictions independently of any new inputs such as real data or other estimation models running simultaneously. DA is a method used in fields such as weather forecasting, which allows for real-time data and information from different models to be incorporated together to estimate the current state of the system. DA models are continuously recalibrated hence do not suffer increasing uncertainty overtime, unlike typical stochastic models. This research aims to take pedestrian movements acquired through a Grand Central Station video feed and produce a model to predict future motion. This model will use data assimilation of both real-time data and a standard trained LSTM RNN developed in collaboration with Dr Minh Kieu.

## File Descriptions

**Ray_DA** = DA Model which uses Particle Filter + LSTM Model + Real World Input (every t = 25,50,75..). Written by Raynil Laxmidas. A flow chart showing how the model functions is shown below. The particle filter used in this model was adapted from [Chapter 12 of Kalman and Bayesian Filters in-Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/24b9fb3cf756b3c765579decd624132efe7be374/12-Particle-Filters.ipynb).

![Flowchart of DA Model using PF + LSTM Model + Real World Input](https://github.com/raylaxmidas/2020-2021-Summer-Research/blob/main/DA%20Model%20Flow%20Chart.JPG)

**Ray_LSTM** = Model which only uses a LSTM model to make pedestrain position predictions. Written by Raynil Laxmidas. 

**Ray_Random** = Model which uses a gaussian distribution to randomly select pedestrian speed and angle in order to make pedestrain position predictions. Written by Raynil Laxmidas. Note that this script was adapted from the "Ray_LSTM" model so makes many references to "LSTM" in variable names etc. however angle and speed prediction are purely created from generated from a gaussian distribution, not the LSTM model.

**M01_LSTM_MK** = Script used to make the LSTM Model. Written by Dr Minh Kieu.

**model** = folder contains the various LSTM Models produced by the M01_LSTM_MK script. Required as a input for Ray_DA and Ray_LSTM.

**processed_data** = folder contains the processed video frames from Grand Central Station in .csv files. These files are required as inputs for all the scripts in this project.

## Visualizations Folder
This folder contains visualisations for the different models predicting two examples cases being PedID = 7 and PedID = 208. PedID = 7 is a relatively simple case, while PedID = 208 is a more complex case where the pedestrian changes their mind about their exit gate resulting in a more complex path. Their actual paths are shown here:

**PedID 07 and 208**

<img src="https://github.com/raylaxmidas/2020-2021-Summer-Research/blob/main/visualizations/agent_7.dat_simple.png" width="400"/><img src="https://github.com/raylaxmidas/2020-2021-Summer-Research/blob/main/visualizations/agent_208.dat_complex.png" width="400"/> 

All the models have visualizations of the paths they have predicted for PedID 07 and 208. Also contained in the visualisations is the rolling performance of each model as they step through time. The DA models which make use of particle filters also have GIFs which illustrate how they function. A few of the key visualizations are shown below:

### PedID = 7 Path Prediction

<img src="https://github.com/raylaxmidas/2020-2021-Summer-Research/blob/main/visualizations/DA%20Model%20(PF%20%2B%20LSTM)/Predicted%20Paths/Ped7/With%20Particles/Particles%20Transparent/Path_GateID_1_PedID_7_Iters_350.jpeg" width="300"/><img src="https://github.com/raylaxmidas/2020-2021-Summer-Research/blob/main/visualizations/DA%20Model%20(PF%20%2B%20Random%20Walk)/Predicted%20Paths/Ped7/With%20Particles/Path_GateID_1_PedID_7_Iters_350.jpeg" width="300"/><img src="https://github.com/raylaxmidas/2020-2021-Summer-Research/blob/main/visualizations/LSTM%20Only%20Model/Predicted%20Paths/Ped7/Path_GateID_1_PedID_7_Iters_350.jpeg" width="300"/><img src="https://github.com/raylaxmidas/2020-2021-Summer-Research/blob/main/visualizations/Random%20Walk%20Model/Predicted%20Paths/Ped7/Path_GateID_1_PedID_7_Iters_350.jpeg" width="300"/>  

## Acknowledgments

I would like to express my acknowledgment towards my supervisors Dr Minh Kieu and Dr Andrea Raith for mentoring myself throughout the course of the summer. Their support made this project very enjoyable to complete.
