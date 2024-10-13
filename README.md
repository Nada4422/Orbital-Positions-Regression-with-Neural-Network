# Orbital Positions Regression with Neural Network
## Objective
The purpose of this task is to study the relationship between orbital positions and time using polynomial regression, implemented through a neural network using the Keras library. The goal is to analyze and visualize the patterns in the data, train a model to predict orbital positions from time steps, and evaluate its performance using various metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²).

## Dataset Description
The dataset used in this project consists of two columns:

1- time_steps: A sequence of time steps representing points in time.

2- y: Corresponding orbital positions at each time step.

The dataset contains both time and position data and may have missing values that are handled by filling with column means. The data is normalized before training the neural network.

## Steps to Run the Code in Google Colab
**1. Upload the Dataset**

1- Ensure you have your dataset file orbit.csv ready to upload in Colab.

2- Modify the path in the code to read the dataset from the Colab environment.

**2. Install Dependencies**

If running in Google Colab, the essential libraries are already installed, but you may want to confirm or reinstall:

!pip install numpy pandas matplotlib scikit-learn tensorflow

**3. Import Libraries**

Make sure to include the necessary Python libraries in the code, as demonstrated below:

import numpy as np

import pandas as pd

import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

**4. Upload the Dataset in Colab**

Run the following code in Colab to upload your dataset:


from google.colab import files

uploaded = files.upload()

Then, replace the file path in the script with the following:

data = pd.read_csv("orbit.csv")

**5. Train the Neural Network**

1- The neural network consists of 4 hidden layers with ReLU activation.

2- You can adjust the number of epochs or other hyperparameters as required for better performance.

**6. Visualize Results**

The script includes two types of visualizations:

1- Line plot comparing actual vs predicted orbits.
2- Scatter plot with predicted and actual positions.

**7. Evaluate the Model**

After training, the model evaluates its performance using MSE, MAE, and R² metrics.

The R² score should aim to be at least +0.75 to demonstrate a good fit.

## Dependencies
To run this project, you need the following libraries:

numpy (1.21.6 or higher)

pandas (1.3.5 or higher)

tensorflow (2.9.0 or higher)

scikit-learn (1.0.2 or higher)

matplotlib (3.5.1 or higher)

**Instructions for Installing Dependencies (if required)**

If running the project on a local machine (not in Colab), use the following command to install the required libraries:

pip install numpy pandas matplotlib scikit-learn tensorflow