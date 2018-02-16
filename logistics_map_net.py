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
def baseline_model():

    the_model = Sequential()
    the_model.add(Dense(30, activation='sigmoid', input_dim=1))
    the_model.add(Dense(1))

   # inputs = Input(shape=(1,))
    #x = Dense(30, activation='sigmoid')(inputs)
    #predictions = Dense(1, activation='sigmoid')(x)
    #model = Model(inputs=inputs, outputs=predictions)

    the_model.summary()
    the_model.compile(optimizer='rmsprop',
                  loss='mean_squared_error')
    return the_model


# Do the work
model = baseline_model()
results = model.fit(X, Y, epochs=1000, batch_size=2, verbose=1)
print("Training is complete.\n")

pyplot.subplot(2, 1, 1)
pyplot.title("Error")
pyplot.plot(results.history['loss'])

# Make predictions

#newdata = numpy.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
newdata = numpy.random.rand(1,100)[0]

prediction = model.predict(newdata, batch_size=1, verbose=1)
print("The prediction is:{0}".format(prediction))



# Plot the predicted and expected results
expected = newdata * 4.0 * (1.0 - newdata)

pyplot.subplot(2, 1, 2)
pyplot.title("Results")
pyplot.scatter(newdata, prediction, s=1, color="red")
pyplot.scatter(newdata, expected, s=1, color="blue")
pyplot.show()
