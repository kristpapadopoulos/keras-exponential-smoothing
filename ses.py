#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

class SES(tf.keras.layers.Layer):
    """
    Tensorflow/Keras implementation of Simple Exponential Smoothing.

    Layer define to learn parameters: alpha
    as per component form of simple exponential smoothing
    defined from Forecasting: Principles and Practice
    by Hyndman and George Athanasopoulos https://otexts.com/fpp2/

    Reference: https://github.com/mcskinner/ets

    Author:    Krist Papadopoulos
    V0 Date:   July 6, 2019
               tensorflow==1.10.1
    V1 Date:   June 14, 2022
               tensorflow==2.6.5
    """
    def __init__(self, min_constraint, max_constraint, dtype=tf.float32):
        super(SES, self).__init__()
        self.min_constraint = min_constraint
        self.max_constraint = max_constraint
    
    def build(self, input_shape):
        self.alpha = self.add_weight('alpha', 
                                     shape=[1,], 
                                     initializer=tf.keras.initializers.random_uniform(0,1),
                                     regularizer=tf.keras.regularizers.L2(0.1),
                                     constraint=tf.keras.constraints.min_max_norm(self.min_constraint,
                                                                                  self.max_constraint))
        
    def call(self, timeseries):
        
        def ses(y, alpha, level):
            '''Apply simple exponential smoothing'''
            forecast = level
            updated_level = forecast + alpha * (y - forecast)
            return forecast, updated_level
        
        predictions = []
        level = tf.reshape(timeseries[0], 1)
        for time_step in timeseries[1:]:
            prediction, level = ses(time_step, self.alpha, level)
            predictions.append(prediction)
        return tf.concat(predictions, -1)
