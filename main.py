#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from ses import SES

# define mean squared loss function for simple exponential smoothing
def ses_loss(prediction, y):
    loss = tf.losses.mean_squared_error(y_true=y, y_pred=prediction)
    return tf.reshape(loss, 1)

if __name__ == "__main__":
    
    #define input series to be learned e.g. log with noise
    y = np.log(np.arange(1,300,3))+np.random.normal(0,0.4,100)
    
    #define tensor of timeseries
    timeseries = tf.convert_to_tensor(y)
    
    #training parameters
    training_epochs = 5
    learning_rate = 0.01
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    loss_history = []
    
    #create ses layer
    ses_layer = SES(min_constraint=0.01, max_constraint=0.5)

    print('--------------------- SES training loss --------------------')
    for epoch in range(training_epochs):
        with tf.GradientTape() as tape:
            prediction = ses_layer(timeseries)
            loss = ses_loss(prediction, timeseries[:-1])

        loss_history.append(loss.numpy())
        grads = tape.gradient(loss, ses_layer.trainable_weights)
        optimizer.apply_gradients(zip(grads, ses_layer.trainable_weights))

        if epoch % 5 == 0:
            print("MSE Loss at epoch {}: {:.3f}, alpha: {:.3f}".format(epoch, loss[0], ses_layer.weights[0].numpy()[0]))

    print("Final MSE loss: {:.3f}".format(loss[0]))
    print("alpha = {:.3f}".format(ses_layer.weights[0].numpy()[0]))
        
    pred = prediction.numpy()
    
    plt.figure(figsize=(15,8))
    plt.title('Simple exponential smoothing layer results with constrained learned parameter')
    plt.plot(list(range(99)), pred, label='pred')
    plt.plot(list(range(99)), timeseries[:-1], label='actual')
    plt.grid(True)
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('amplitude')
