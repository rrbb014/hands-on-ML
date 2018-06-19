# Linear Regression with Tensorflow p.303 - 304
# Analytically compute

import numpy as np

import tensorflow as tf
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

# optimal theta = (X^T dot X)^-1 X^T dot y
# https://en.wikipedia.org/wiki/Linear_least_squares_(mathematics)

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name='X')  # shape=(20640, 9)
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y') # Reshape 2d array, (20640, 1)
XT = tf.transpose(X) # shape= (9, 20640)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)   
# XT dot X and inverse -> (9, 9)
# above matrix and XT -> 9, 20640
# above and y -> 9, 1


with tf.Session() as sess:
	theta_value = theta.eval()
	theta_T = tf.transpose(theta_value)
	y_pred = tf.matmul(theta_T, XT)
	loss = tf.subtract(y_pred, tf.transpose(y), name='loss')
	loss_value = loss.eval()

# theta_T dot XT -> 1, 20640 => y_pred
 
from IPython import embed
embed()
print('Summation of Loss : ', np.sum(loss_value))


