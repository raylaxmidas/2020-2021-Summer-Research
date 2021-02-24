# 2020-2021-Summer-Research
Dynamic Agent-Based Modelling with Data Assimilation in Urban Areas

**Ray_DA** = DA Model which uses Particle Filter + LSTM Model + Real World Input (every t = 25,50,75..). Written by Raynil Laxmidas.

**Ray_LSTM** = Model which only uses a LSTM model to make pedestrain position predictions. Written by Raynil Laxmidas. 

**Ray_Random** = Model which uses a gaussian distribution to randomly select pedestrian speed and angle in order to make pedestrain position predictions. Written by Raynil Laxmidas.

**M01_LSTM_MK** = Script used to make the LSTM Model. Written by Dr Minh Kieu.

**model** = folder contains the various LSTM Models produced by the M01_LSTM_MK script. Required as a input for Ray_DA and Ray_LSTM.

**processed_data** = folder contains the processed video frames from Grand Central Station in .csv files. These files are required as inputs for all the scripts in this project.
