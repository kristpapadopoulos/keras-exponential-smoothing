#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tensorflow/Keras implementation of Simple Exponential Smoothing.

Layer define to learn parameters: alpha and initial level
as per component form of simple exponential smoothing
defined from Forecasting: Principles and Practice
by Hyndman and George Athanasopoulos https://otexts.com/fpp2/

Reference: https://github.com/mcskinner/ets

Author:    Krist Papadopoulos
V0 Date:   July 6, 2019
        
           tensorflow==1.10.1
"""

import tensorflow as tf

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