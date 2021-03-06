# 2020 - 2021 Summer Research, Raynil Laxmidas
**Dynamic Agent-Based Modelling with Data Assimilation in Urban Areas**

<object data="https://github.com/raylaxmidas/2020-2021-Summer-Research/blob/main/SRS%20Poster%20-%20Raynil%20Laxmidas%20-%20v5.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="https://github.com/raylaxmidas/2020-2021-Summer-Research/blob/main/SRS%20Poster%20-%20Raynil%20Laxmidas%20-%20v5.pdf">
        <p>Download the poster for this project here: <a href="https://github.com/raylaxmidas/2020-2021-Summer-Research/blob/main/SRS%20Poster%20-%20Raynil%20Laxmidas%20-%20v5.pdf">Project Poster</a>.</p>
    </embed>
</object>

## 1.0 Objectives:
The objective of this research was to predict pedestrian movements in grand central train station through:

- Creating a **Long short-term memory (LSTM)** artificial recurrent neural network (RNN) that predicts pedestrian angle and movement speed based on 5 previous frames of video data.
- Use a **particle filter (PF)** to incorporate LSTM inputs and real-time data to make predictions of pedestrian movements and the gate at which they will exit the train station.
- Compare the performance of the **data assimilation (DA)** model using PF + LSTM to using a LSTM only model and a **random walk (RW)** model.

## 2.0 Introduction:
Typically, prediction models are trained based on previous historical data and then used to make predictions independently of any new inputs such as real data or other estimation models running simultaneously. Data assimilation is a method used in fields such as weather forecasting, which allows for real-time data and information from different models to be incorporated together to estimate the current state of the system. Data assimilation models are continuously recalibrated; hence do not suffer increasing uncertainty overtime, unlike typical stochastic models. This research aims to take processed pedestrian movements acquired through a Grand Central Station video feed and produce a model to predict future pedestrian motion. This model will use data assimilation via particle filtering of both real-time data and a standard trained long short-term memory recurrent neural network developed in collaboration with Dr Minh Kieu.

## 3.0 File Descriptions:

**Ray_DA** = DA Model which uses Particle Filter + LSTM Model + Real World Input (every t = 25,50,75..). A flow chart showing how the model functions is shown below. The particle filter used in this model was adapted from [Chapter 12 of Kalman and Bayesian Filters in-Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/24b9fb3cf756b3c765579decd624132efe7be374/12-Particle-Filters.ipynb). Written by Raynil Laxmidas.

![Flowchart of DA Model using PF + LSTM Model + Real World Input](https://github.com/raylaxmidas/2020-2021-Summer-Research/blob/main/DA%20Model%20Flow%20Chart.JPG)

**Ray_LSTM** = Model which only uses a LSTM inputs to make pedestrain position predictions. Written by Raynil Laxmidas. 

**Ray_Random** = Model which uses a gaussian distribution to randomly select pedestrian speed and angle in order to make pedestrain position predictions. This script was adapted from the "Ray_LSTM" model so makes many references to "LSTM" in variable names etc. however angle and speed prediction are purely created from generated from a gaussian distribution, not the LSTM model. Written by Raynil Laxmidas.

**M01_LSTM_MK** = Script used to make the LSTM Model. Written by Dr Minh Kieu.

**model** = folder contains the various LSTM Models produced by the M01_LSTM_MK script. Required as a input for Ray_DA and Ray_LSTM.

**meta_analysis** = contains the raw preformance outputs used to generate the meta-analysis.

**processed_data** = folder contains the processed video frames from Grand Central Station in .csv files. These files are required as inputs for all the scripts in this project.

**SRS Poster - Raynil Laxmidas - v5.pdf** = Poster to accompany project.

## 4.0 Visualizations / Results
This folder contains visualisations for the different models predicting two examples cases being PedID = 7 and PedID = 208. PedID = 7 is a relatively simple case, while PedID = 208 is a more complex case where the pedestrian changes their mind about their exit gate resulting in a more complex path. Their actual paths are shown here:

**PedID 07 and 208**

<img src="https://github.com/raylaxmidas/2020-2021-Summer-Research/blob/main/visualizations/agent_7.dat_simple.png" width="400"/><img src="https://github.com/raylaxmidas/2020-2021-Summer-Research/blob/main/visualizations/agent_208.dat_complex.png" width="400"/> 

All the models have visualizations of the paths they have predicted for PedID 07 and 208. Also contained in the visualisations is the rolling performance of each model as they step through time. The DA models which make use of particle filters also have GIFs which illustrate how they function. A few of the key visualizations are shown below:

*(Note: the non-DA models have been given the correct exit gates in order to form their path predictions. These models actually produce 10 prediction for each possible exit gate, plots of these can be found in visualizations folder.)*

### 4.1 PedID 7 Path Predictions:
DA models receive real world data every 25 time steps for re-calibration and have been initialized with 100 particles.

<img src="https://github.com/raylaxmidas/2020-2021-Summer-Research/blob/main/visualizations/DA%20Model%20(PF%20%2B%20LSTM)/Predicted%20Paths/Ped7/No%20Particles/Path_GateID_1_PedID_7_Iters_350.jpeg" width="400"/><img src="https://github.com/raylaxmidas/2020-2021-Summer-Research/blob/main/visualizations/DA%20Model%20(PF%20%2B%20Random%20Walk)/Predicted%20Paths/Ped7/No%20Particles/Path_GateID_1_PedID_7_Iters_350.jpeg" width="400"/><img src="https://github.com/raylaxmidas/2020-2021-Summer-Research/blob/main/visualizations/LSTM%20Only%20Model/Predicted%20Paths/Ped7/Path_GateID_1_PedID_7_Iters_350.jpeg" width="400"/><img src="https://github.com/raylaxmidas/2020-2021-Summer-Research/blob/main/visualizations/Random%20Walk%20Model/Predicted%20Paths/Ped7/Path_GateID_1_PedID_7_Iters_350.jpeg" width="400"/>  

### PedID 7 DA Model (PF + LSTM) Animation:
<img src="https://github.com/raylaxmidas/2020-2021-Summer-Research/blob/main/visualizations/DA%20Model%20(PF%20%2B%20LSTM)/GIFs%20of%20Particle%20Filter%20Running/Ped7/Version%201.gif" width="500"/>

### 4.2 Model Benchmarking of PedID 7
The following figure shows the error in pixels throughout each time step predicted for all four models (Note the plot is on a log scale). It can be seen the data assimilation models are able to continuously re-calibrate and hence do not suffer increasing uncertainty unlike the other models. Data Assimilation models had an average error about 10 pixels per iteration which was significantly better than the LSTM only and random walk models which had an average error of 367 and 351 pixels per iteration respectively for PedID 7.

<img src="https://github.com/raylaxmidas/2020-2021-Summer-Research/blob/main/visualizations/Performance%20Comparison.JPG" width="800"/>

### 4.3 PedID 208 Path Predictions:
DA models receive real world data every 25 time steps for re-calibration and have been initialized with 100 particles.

<img src="https://github.com/raylaxmidas/2020-2021-Summer-Research/blob/main/visualizations/DA%20Model%20(PF%20%2B%20LSTM)/Predicted%20Paths/Ped208/No%20Particles/Path_GateID_9_PedID_208_Iters_1000.jpeg" width="400"/><img src="https://github.com/raylaxmidas/2020-2021-Summer-Research/blob/main/visualizations/DA%20Model%20(PF%20%2B%20Random%20Walk)/Predicted%20Paths/Ped208/No%20Particles/Path_GateID_9_PedID_208_Iters_1000.jpeg" width="400"/><img src="https://github.com/raylaxmidas/2020-2021-Summer-Research/blob/main/visualizations/LSTM%20Only%20Model/Predicted%20Paths/Ped208/Path_GateID_9_PedID_208_Iters_1000.jpeg" width="400"/><img src="https://github.com/raylaxmidas/2020-2021-Summer-Research/blob/main/visualizations/Random%20Walk%20Model/Predicted%20Paths/Ped208/Path_GateID_9_PedID_208_Iters_1000.jpeg" width="400"/> 

### 4.4 PedID 208 DA Model (PF + LSTM) Animation:
<img src="https://github.com/raylaxmidas/2020-2021-Summer-Research/blob/main/visualizations/DA%20Model%20(PF%20%2B%20LSTM)/GIFs%20of%20Particle%20Filter%20Running/Ped208/gate9_compressed.gif" width="500"/>

## 5.0 Meta - Analysis of Models
A meta-analysis of the four models was performed against all pedestrians present in the video frame at time = 1010, which had sufficient data points to run 300 iterations (time step predictions) of the models. Of the 47 pedestrians, 14 were excluded as they had less than 300 data points. 

Data assimilation models were run with 1000 particles and were recalibrated every 25 iterations. All models performed 300 predictions into the future. As the non - DA models cannot determine the gate through calibration inputs, all models were provided with the true gate as an input. 

It can be seen that the DA models preform considerably better than the non - DA models. The best performing model was the DA model using PF and LSTM. This model performed 1.3, 9.4 and 12.5 times better than the DA model using PF and Random Walk, LSTM Only and Random Walk Only models respectively. 

Raw data for this meta analysis can be found in the "meta_analysis" folder.

<img src="https://github.com/raylaxmidas/2020-2021-Summer-Research/blob/main/meta_analysis/Meta-Analysis%20Linear.png" width="800"/>

## 6.0 Acknowledgments:

I would like to express my acknowledgment towards my supervisors Dr Minh Kieu and Dr Andrea Raith for mentoring me throughout the summer. Their support made this project very enjoyable to complete.

## 7.0 References:
Malleson, N., Minors, K., Kieu, L.-M., Ward, J. A., West, A., & Heppenstall, A. (2020). Simulating Crowds in Real Time with Agent-Based Modelling and a Particle Filter. Journal of Artificial Societies and Social Simulation, 23(3), 3.

Srinivasan, S. (2019, August 14). Particle Filter: A hero in the world of Non-Linearity and Non-Gaussian. Medium. https://towardsdatascience.com/particle-filter-a-hero-in-the-world-of-non-linearity-and-non-gaussian-6d8947f4a3dc

Rlabbe/Kalman-and-Bayesian-Filters-in-Python. (n.d.). GitHub. Retrieved February 24, 2021, from https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

