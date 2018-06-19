# gradient based optimization p.305 - 
# Auto Differentiation

import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

AUTO_GRADIENT = True

if __name__ == '__main__':
	# Data Preprocessing
	scaler = StandardScaler()
	housing = fetch_california_housing()
	m, n = housing.data.shape
	scaled_housing = scaler.fit_transform(housing.data)
	scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing]

	# Test code preprocessed data dimension 
	assert scaled_housing_data_plus_bias.shape == (m, n+1)


	n_epochs = 100000
	learning_rate = 1e-3

	X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name='X') # 20640, 9
	y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
	
	# Create weight parameter n+1 * 1 vector from min value -1 to max value +1 in uniform distribution  
	theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0), name='theta')
	y_pred = tf.matmul(X, theta, name='predictions')
	error = y_pred - y  # 20640, 1
	mse = tf.reduce_mean(tf.square(error), name='mse')
	
	
	if AUTO_GRADIENT:
		# Automatically compute gradient
		gradients = tf.gradients(mse, [theta])[0]
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
		training_op = optimizer.minimize(mse)

	else:
		# Analytically compute gradient 
		gradients = 2/m * tf.matmul(tf.transpose(X), error) # 9, 1
	
		training_op = tf.assign(theta, theta - learning_rate * gradients)
	
	init = tf.global_variables_initializer()
	
	with tf.Session() as sess:
		sess.run(init)
		
		for epoch in range(n_epochs):
			if epoch % 1000 == 0:
				print('Epoch ', epoch, 'MSE : ', mse.eval())
			sess.run(training_op)
		
		best_theta = theta.eval()
		