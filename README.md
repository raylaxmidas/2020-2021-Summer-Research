# 2020-2021-Summer-Research
Dynamic Agent-Based Modelling with Data Assimilation in Urban Areas

The objective of this research was to predict pedestrian movements in grand central train station through:

- Creating a **Long short-term memory (LSTM)** artificial recurrent neural network (RNN) that predicts pedestrian angle and movement speed based on 5 previous frames of video data.
- Use a **particle filter (PF)** to incorporate LSTM inputs and real-time data to make predictions of pedestrian movements and the gate at which they will exit the train station.
- Compare the performance of the **data assimilation (DA)** model using PF + LSTM + Real Data to using a LSTM only model and a random walk model.

Typically, prediction models are trained based on previous historical data and then used to make predictions independently of any new inputs such as real data or other estimation models running simultaneously. DA is a method used in fields such as weather forecasting, which allows for real-time data and information from different models to be incorporated together to estimate the current state of the system. DA models are continuously recalibrated hence do not suffer increasing uncertainty overtime, unlike typical stochastic models. This research aims to take pedestrian movements acquired through a Grand Central Station video feed and produce a model to predict future motion. This model will use data assimilation of both real-time data and a standard trained LSTM RNN developed in collaboration with Dr Minh Kieu.

## File Descriptions

**Ray_DA** = DA Model which uses Particle Filter + LSTM Model + Real World Input (every t = 25,50,75..). Written by Raynil Laxmidas. A flow chart showing how the model functions is shown below.

![Flowchart of DA Model using PF + LSTM Model + Real World Input](https://github.com/raylaxmidas/2020-2021-Summer-Research/blob/main/DA%20Model%20Flow%20Chart.JPG)

**Ray_LSTM** = Model which only uses a LSTM model to make pedestrain position predictions. Written by Raynil Laxmidas. 

**Ray_Random** = Model which uses a gaussian distribution to randomly select pedestrian speed and angle in order to make pedestrain position predictions. Written by Raynil Laxmidas.

**M01_LSTM_MK** = Script used to make the LSTM Model. Written by Dr Minh Kieu.

**model** = folder contains the various LSTM Models produced by the M01_LSTM_MK script. Required as a input for Ray_DA and Ray_LSTM.

**processed_data** = folder contains the processed video frames from Grand Central Station in .csv files. These files are required as inputs for all the scripts in this project.
