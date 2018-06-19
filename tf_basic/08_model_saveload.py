# Model Save & Load

import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler


EPOCHS = 100
LEARNING_RATE = 1e-3
BATCH_SIZE = 100

def fetch_batch(epoch, batch_index, batch_size):
	# load data from disk
	np.random.seed(epoch*n_batches + batch_index)
	indices = np.random.randint(m, size=batch_size)
	X_batch = scaled_housing_data_plus_bias[indices]
	y_batch = housing.target.reshape(-1, 1)[indices]
	return X_batch, y_batch

if __name__ == '__main__':
	# Data Preprocessing
	scaler = StandardScaler()
	housing = fetch_california_housing()
	m, n = housing.data.shape
	scaled_housing = scaler.fit_transform(housing.data)
	scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing]

	# Test code preprocessed data dimension 
	assert scaled_housing_data_plus_bias.shape == (m, n+1)
	
	# Changing point (constant -> placeholder)
	X = tf.placeholder(tf.float32, shape=(None, n+1), name='X')
	y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
	
	theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0), name='theta')
	y_pred = tf.matmul(X, theta, name='predictions')
	error = y_pred - y
	mse = tf.reduce_mean(tf.square(error), name='mse')
	
	gradients = tf.gradients(mse, [theta])[0]
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
	training_op = optimizer.minimize(mse)
	
	n_batches = int(np.ceil(m / BATCH_SIZE))
	init = tf.global_variables_initializer()
	
	# Model Saver initialization
	saver = tf.train.Saver({'weights': theta})
	with tf.Session() as sess:
		#sess.run(init)
		# Model Load
		saver.restore(sess, "./tmp/my_model_final.ckpt")		
		for epoch in range(EPOCHS):
			if epoch % 100 == 0:
				# Save Model
				save_path = saver.save(sess, "./tmp/my_model.ckpt")
			for batch_index in range(n_batches):
				X_batch, y_batch = fetch_batch(epoch, batch_index, BATCH_SIZE)
				sess.run(training_op, feed_dict={X: X_batch, y:y_batch})
		
		best_theta = theta.eval()
		save_path = saver.save(sess, "./tmp/my_model_final.ckpt")
		
	print(best_theta)