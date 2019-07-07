#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example training of Keras Simple Exponential Smoothing Implementation


Author:    Krist Papadopoulos
V0 Date:   July 6, 2019
        
           tensorflow==1.10.1
           numpy==1.14.5 

"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.enable_eager_execution()

class SES(tf.keras.layers.Layer):
    def __init__(self, dtype=tf.float32):
        super(SES, self).__init__()
    
    def build(self, input_shape):
        self.alpha = self.add_weight('alpha', shape=[1,], 
                                     initializer=tf.keras.initializers.random_uniform(0,1), 
                                     constraint=tf.keras.constraints.min_max_norm(0,1))
        
        self.level = self.add_weight('level', shape=[1,], 
                                     initializer=tf.keras.initializers.truncated_normal())
        
    def call(self, input):
        
        def ses(y, alpha, level):
            '''Simple exponential smoothing using component form
            from Forecasting: Principles and Practice - Hyndman and George Athanasopoulos'''
            forecast = level
            updated_level = forecast + alpha * (y - forecast)
            return forecast, updated_level
        
        predictions = []
        for time_step in range(input.shape[0]):
            prediction, self.level = ses(input[time_step], self.alpha, self.level)
            predictions.append(prediction)
         
        return tf.concat(predictions, axis=-1)

# define mean squared loss function for simple exponential smoothing
def ses_loss(prediction, y):
    loss = tf.losses.mean_squared_error(labels=y, predictions=prediction, weights=1)
    return loss

if __name__ == "__main__":
    
    #define input series to be learned e.g. log with noise
    y = np.log(np.arange(1,300,3))+np.random.normal(0,0.6,100)
    
    #define tensorflow dataset
    y_values = tf.data.Dataset.from_tensor_slices(y).batch(batch_size=y.shape[0])
    
    #training parameters
    training_epochs = 500
    learning_rate = 0.01
    optimizer = tf.train.AdamOptimizer(learning_rate)

    loss_history = []
    
    #call SES layer
    ses_layer = SES()

    print('--------------------- SES training loss --------------------')
    for epoch in range(training_epochs):
        for yi in y_values:
            with tf.GradientTape() as tape:
                prediction = ses_layer(yi)
                loss = ses_loss(prediction, yi)

            loss_history.append(loss.numpy())
            grads = tape.gradient(loss, ses_layer.trainable_weights)
            optimizer.apply_gradients(zip(grads, ses_layer.trainable_weights), 
                                      global_step=None)
        
            if epoch % 20 == 0:
                print("Loss at step {:03d}: {:.3f}, alpha: {:.3f}, initial level: {:.3f}".format(epoch, loss, 
                      ses_layer.weights[0].numpy()[0],
                      ses_layer.weights[1].numpy()[0]))

    print("Final loss: {:.3f}".format(loss))
    print("alpha = {:.3f}, initial level = {:.3f}".format(ses_layer.weights[0].numpy()[0], 
          ses_layer.weights[1].numpy()[0]))
    
    plt.title('SES of Input Series')
    plt.plot(y)
    plt.plot(prediction.numpy())
    plt.grid(True)