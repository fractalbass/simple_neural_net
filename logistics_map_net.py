# *****************************************
# A simple neural net using
# Keras and TensorFlow for
# training a neural netowrk to learn
# the Feigenbaum Map
#
# June 1, 2017
# Miles R. Porter, Painted Harmony Group
# This code is free to use and distribute
# *****************************************
import tensorflow as tf
import numpy as np
import os
np.random.seed(42)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#from tensorflow.random import set_random_seed
#set_random_seed(42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime



# load dataset
df = pd.read_csv("logistics_10k.csv", header=1, names=['n','n+1'])


# split into input (X) and output (Y) variables

X = df['n']
Y = df['n+1']


# Set up the network

def neural_network_model():

    the_model = Sequential()
    the_model.add(Dense(5, activation='relu', input_dim=1))
    the_model.add(Dense(500, activation='relu', input_dim=1))
    the_model.add(Dense(1200, activation='relu', input_dim=1))
    the_model.add(Dense(500, activation='relu', input_dim=1))
    the_model.add(Dense(5, activation='relu', input_dim=1))
    the_model.add(Dense(1))
    the_model.summary()
    the_model.compile(optimizer='rmsprop',
                  loss='mean_squared_error')
    return the_model


model = neural_network_model()
st = datetime.now()
results = model.fit(X, Y, epochs=25, batch_size=16, verbose=1)
et = datetime.now()
print("Training is complete.\n")
print("\n\nTraining time: {}\n\n".format(et-st))

plt.subplot(2, 1, 1)
plt.title("Error")
plt.plot(results.history['loss'])

# Make predictions

inputs = np.random.rand(1, 100)[0]

prediction = model.predict(inputs, batch_size=1, verbose=0)

# Plot the predicted and expected results
expected = inputs * 4.0 * (1.0 - inputs)

plt.subplot(2, 1, 2)
plt.title("Results")
for i in range(0, len(inputs)):
    p = prediction[i]
    e = expected[i]
    plt.scatter(i, p, s=1, color="red")
    plt.scatter(i, e, s=1, color="blue")

plt.show()
