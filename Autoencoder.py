# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 15:43:05 2018

@author: hasee
"""
import math
import os
import sys
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot = False)
class Classifier:
    def __init__(self, batch_size, learning_rate, training_epochs, dropout_rate
                 , batch_norm_use):

        """
        Initializes a VGG Classifier architecture
        :param batch_size: The size of the data batch
        :param layer_stage_sizes: A list containing the filters for each layer stage, where layer stage is a series of
        convolutional layers with stride=1 and no max pooling followed by a dimensionality reducing stage which is
        either a convolution with stride=1 followed by max pooling or a convolution with stride=2
        (i.e. strided convolution). So if we pass a list [64, 128, 256] it means that if we have inner_layer_depth=2
        then stage 0 will have 2 layers with stride=1 and filter size=64 and another dimensionality reducing convolution
        with either stride=1 and max pooling or stride=2 to dimensionality reduce. Similarly for the other stages.
        :param name: Name of the network
        :param num_classes: Number of classes we will need to classify
        :param num_channels: Number of channels of our image data.
        :param batch_norm_use: Whether to use batch norm between layers or not.
        :param inner_layer_depth: The amount of extra layers on top of the dimensionality reducing stage to have per
        layer stage.
        :param strided_dim_reduction: Whether to use strided convolutions instead of max pooling.
        """
        self.batch_size = batch_size
        self.batch_norm_use = batch_norm_use
        self.weights = {}
        self.biases = {}
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs
        self.dropout_rate = dropout_rate
#        self.c = 0

    def __call__(self, input_data, display_step, n_input):
        """
        Runs the CNN producing the predictions and the gradients.
        :param input_data: Matrix input for training. e.g. for sushi data [users, item features]
        :param n_input: Dimensionality of input
        :param dropout_rate: A tf placeholder of type tf.float32 indicating the amount of dropout applied
        :return: Autoencoder Result
        """
        
        
        X = tf.placeholder("float",[None,n_input])

        # hidden layer settings
        n_hidden_1 = 4 # 1st layer num features
        n_hidden_2 = 2 # 2nd layer num features
        self.weights = {
        	'encoder_h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
        	'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
        	'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2,n_hidden_1])),
        	'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
        	}
        self.biases = {
        	'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        	'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
        	'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        	'decoder_b2': tf.Variable(tf.random_normal([n_input])),
        	}
        
          # Construct model
        encoder_op = self.encoder(X) 			# 2 Features
        decoder_op = self.decoder(encoder_op)	# 8 Features
    
        # Prediction
        y_pred = decoder_op	# After 
        # Targets (Labels) are the input data.
        y_true = X			# Before
        
        # Define loss and optimizer, minimize the squared error
        cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)
    
        
    
        # Launch the graph
        with tf.Session() as sess:
            # tf 马上就要废弃tf.initialize_all_variables()这种写法
            # 替换成下面:
            sess.run(tf.global_variables_initializer())
        #    total_batch = int(mnist.train.num_examples/batch_size)
        #    total_batch = int(data.shape[0]/batch_size)
            # Training cycle
            for epoch in range(self.training_epochs):
                # Loop over all batches
                for start in range(0, input_data.shape[0]-self.batch_size, self.batch_size):
                    end = start + self.batch_size
#                zip(
#                        range(0, input_data.shape[0], self.batch_size),
#                        range(self.batch_size, input_data.shape[0], self.batch_size)
#                        ):
                    batch_xs= input_data[start:end,:]
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
                # Display logs per epoch step
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))
        
            print("Optimization Finished!")
            
            encoder_result = sess.run(encoder_op, feed_dict={X: input_data})
#            print ("weight")
#            print (sess.run(self.weights['encoder_h1']))
#            print (sess.run(self.weights['encoder_h2']))
#            print (encoder_result)
#            print (input_data)


        return encoder_result

    def encoder(self,x):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['encoder_h1']),
                                       self.biases['encoder_b1']))
        # Decoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['encoder_h2']),
                                       self.biases['encoder_b2']))
        return layer_2
        
    # Building the decoder
    def decoder(self,x):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['decoder_h1']),
                                       self.biases['decoder_b1']))
        # Decoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['decoder_h2']),
                                       self.biases['decoder_b2']))
        return layer_2
    
  
    








