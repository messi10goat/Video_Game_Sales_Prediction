# messi10goat.github.io
# Videogames earning projections
"""
Parameters :
    1. Critic Rating
    2. Genre
    3. Exclusivity of the deal
    4. Portability of the game
    5. Is a sequel of an earlier video game
    6. Suitable for kids
    7. Unit Price
    
Prediction:
    Total Earnings
    
    
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


traindata = pd.read_csv("sales_data_training.csv", dtype = float)
xtrain = traindata.drop('total_earnings', axis=1).values
ytrain = traindata[['total_earnings']].values



testdata = pd.read_csv("sales_data_training.csv", dtype = float)
xtest = testdata.drop('total_earnings', axis=1).values
ytest = testdata[['total_earnings']].values

xscaler = MinMaxScaler(feature_range=(0,1))
yscaler = MinMaxScaler(feature_range=(0,1))

xtrainscaled = xscaler.fit_transform(xtrain)
ytrainscaled = yscaler.fit_transform(ytrain)

xtestscaled = xscaler.transform(xtest)
ytestscaled = yscaler.transform(ytest)

ymultiplier = yscaler.scale_[0]
yadder = yscaler.min_[0]

#########################

learning_rate = 0.001
training_epochs = 100
display_step = 5

number_of_inputs = 9
number_of_outputs = 1

n1 = 50
n2 = 100
n3 = 50

with tf.compat.v1.variable_scope('input'):
    X = tf.compat.v1.placeholder(tf.float32, shape=(None, number_of_inputs))

with tf.compat.v1.variable_scope('layer_1'):
    weights = tf.get_variable(name = "weights1", shape=[number_of_inputs, n1], initializer = tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name = "biases1", shape=[n1], initializer = tf.zeros_initializer())
    layer_1_output = tf.nn.relu(tf.matmul(X,weights)+biases)
    
with tf.compat.v1.variable_scope('layer_2'):
    weights = tf.get_variable(name = "weights2", shape=[n1, n2], initializer = tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name = "biases2", shape=[n2], initializer = tf.zeros_initializer())
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output,weights)+biases)
    
with tf.compat.v1.variable_scope('layer_3'):
    weights = tf.get_variable(name = "weights3", shape=[n2, n3], initializer = tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name = "biases3", shape=[n3], initializer = tf.zeros_initializer())
    layer_3_output = tf.nn.relu(tf.matmul(layer_2_output,weights)+biases)
    
with tf.compat.v1.variable_scope('output'):
    weights = tf.get_variable(name = "weights4", shape=[n3, number_of_outputs], initializer = tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name = "biases4", shape=[number_of_outputs], initializer = tf.zeros_initializer())
    prediction = tf.nn.relu(tf.matmul(layer_3_output,weights)+biases)
    
with tf.compat.v1.variable_scope('cost'):
    Y = tf.compat.v1.placeholder(tf.float32, shape=(None,1))
    cost = tf.reduce_mean(tf.squared_difference(prediction,Y))
    
with tf.compat.v1.variable_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        session.run(optimizer,feed_dict={X : xtrainscaled, Y:ytrainscaled})
        if epoch%5==0:
            training_cost=session.run(cost, feed_dict={X : xtrainscaled, Y:ytrainscaled} )
            test_cost=session.run(cost, feed_dict={X : xtestscaled, Y:ytestscaled} )
            print("epoch = ",epoch,"  training cost = ",training_cost, " test cost = ",test_cost)
        
    print("training is complete")
        
        
    
    
    
    