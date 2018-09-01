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


class Autoencoder:
    def __init__(self, batch_size, learning_rate, training_epochs, dropout_rate
                 , batch_norm_use):

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


        return encoder_result

    def encoder(self,x):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, self.weights['encoder_h1']),
                                       self.biases['encoder_b1']))
        # Decoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, self.weights['encoder_h2']),
                                       self.biases['encoder_b2']))
        return layer_2
        
    # Building the decoder
    def decoder(self,x):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, self.weights['decoder_h1']),
                                       self.biases['decoder_b1']))
        # Decoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, self.weights['decoder_h2']),
                                       self.biases['decoder_b2']))
        return layer_2
    

class DNN:
    def __init__(self, batch_size, learning_rate, training_epochs, dropout_rate
                 , batch_norm_use):
        
        tf.reset_default_graph()
        self.batch_size = batch_size
        self.batch_norm_use = batch_norm_use
        self.weights = {}
        self.biases = {}
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs
        self.dropout_rate = dropout_rate


    def __call__(self, training_data, item, display_step, n_input, n_output):
        """
        Runs the CNN producing the predictions and the gradients.
        :param input_data: Matrix input for training. e.g. for sushi data [users, item features]
        :param n_input: Dimensionality of input
        :param dropout_rate: A tf placeholder of type tf.float32 indicating the amount of dropout applied
        :return: DNN Result
        """
        target = training_data[:,-1]
        train_data = training_data[0:int(training_data.shape[0]*0.9),0:-1]
        train_label = target[0:int(training_data.shape[0]*0.9)]
#        test_data = training_data[int(training_data.shape[0]*0.9):,0:-1]
#        test_label = target[int(training_data.shape[0]*0.9):]

        X = tf.placeholder("float",[None,n_input])
        y = tf.placeholder("float",[None,n_output])

        # Prediction
        y_pred = self.framework(X)	# After 
        # Targets (Labels) of the data.
        y_true = tf.cast(y,	dtype=tf.int32)		# Before


        
        # Define loss and optimizer, minimize the squared error
#        cost = tf.reduce_mean(tf.square(y_true-y_pred))
        cost= tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred[1], labels=y_true[:,0], name="cross_entropy")
        sum_cost=tf.reduce_mean([cost])
        
#        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(sum_cost)
        
        correction_prediction = tf.equal(tf.argmax(tf.nn.softmax(y_pred[1]),1),tf.cast(y_true, tf.int64))
        accuracy = tf.reduce_mean(tf.cast(correction_prediction,tf.float32))
        
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
                for start in range(0, train_data.shape[0]-self.batch_size, self.batch_size):
                    end = start + self.batch_size
                    
                    batch_xs = train_data[start:end,:]
                    batch_ys = train_label[start:end].reshape([-1,1])

         
                    _,c = sess.run([optimizer,sum_cost], feed_dict={X: batch_xs,y:batch_ys})
                    acc = sess.run(accuracy,feed_dict={X: batch_xs,y:batch_ys})
                # Display logs per epoch step
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c),
                          "acc=", "{:.9f}".format(acc))
 
            print("Optimization Finished!")
            
            output = sess.run(y_pred, feed_dict={X: item})[0]
        return output

    def framework(self,X):
        layer_1 = tf.layers.dense(X, units=16,activation=tf.nn.elu,name='layer1')
        layer_2 = tf.layers.dense(layer_1, units=24,activation=tf.nn.elu,name='layer2')
        layer_3 = tf.layers.dense(layer_2, units=4,activation=tf.nn.elu,name='layer3')
        prediction = tf.layers.dense(layer_3, units=5,name='layer4')
        
        return [layer_3, prediction]

    
        







