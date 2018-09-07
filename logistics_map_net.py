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

import numpy
import pandas
from keras.models import Model, Sequential
from keras.layers import Dense, Input
from matplotlib import pyplot

# load dataset
dataframe = pandas.read_csv("logistics.csv", delim_whitespace=True, header=None)
dataset = dataframe.values

# split into input (X) and output (Y) variables

X = dataset[:, 0]
Y = dataset[:, 1]


# Set up the network

def neural_network_model():

    the_model = Sequential()
    the_model.add(Dense(30, activation='sigmoid', input_dim=1))
    the_model.add(Dense(1))
    the_model.summary()
    the_model.compile(optimizer='rmsprop',
                  loss='mean_squared_error')
    return the_model


model = neural_network_model()
results = model.fit(X, Y, epochs=1000, batch_size=2, verbose=0)
print("Training is complete.\n")

pyplot.subplot(2, 1, 1)
pyplot.title("Error")
pyplot.plot(results.history['loss'])

# Make predictions

inputs = numpy.random.rand(1,100)[0]

prediction = model.predict(inputs, batch_size=1, verbose=0)

# Plot the predicted and expected results
expected = inputs * 4.0 * (1.0 - inputs)

pyplot.subplot(2, 1, 2)
pyplot.title("Results")
for i in range(0, len(inputs)):
    p = prediction[i]
    e = expected[i]
    pyplot.scatter(i, p, s=1, color="red")
    pyplot.scatter(i, e, s=1, color="blue")

pyplot.show()
