# simple_neural_net

This is a simple program that trains a neural network to learn to predict items in the logistics map.  More information
on the logistics map can be found [here](https://en.wikipedia.org/wiki/Feigenbaum_constants).
 
 In order to run this code, you will need a few things:
 
 + Python 3
 + Several Python Libraries (these can be installed with pip):
    + Keras 2.1.6
    + Numpy 1.14.3
    + Pandas 0.23.0
    + Tensorflow 1.6
    + Matplotlib 2.2.2
    
(Other versions may work.  These are just what I am running.)

To run the script, issue the following command at a command prompt:
```python 
python logistics_map_net.py
```

The program will train a simple dense neural network based on data in the logistics.csv file.  The network predicts
the next value in a sequence of random numbers based on the last value in the sequence.  For a discussion of similar
work please refer to [this paper](https://www.osti.gov/biblio/5470451-nonlinear-signal-processing-using-neural-networks-prediction-system-modelling)
.

The final output will display a graph with two panes.  The top pane shows the "learning curve" of the network.  The
bottom pane shows how the network performs on unseen data.  The BLUE dots in the bottom pane represent expected output 
values while the RED dots represent what the network guessed the output should be.

(Lapedes, A., and Farber, R.. Nonlinear signal processing using neural networks: Prediction and system modelling. United States: N. p., 1987. Web.)

Some notes: The program will train a neural network that consists of 3 layers...  one input layer of a single node, a hidden layer 
of 30 nodes and an output layer of 1 single node.  The activation function of the network is a sigmoid function, and the
optimizer function is [RMSProp](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf).  Please refer
to the Keras docs for more information on activation functions and optimizers available through Keras.

For questions or comments, please don't hesitate to reach out to me via email.

Miles Porter

mporter (at) paintedharmony (dot) com
 