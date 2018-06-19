# Modularization

import tensorflow as tf
from datetime import datetime


def relu(X):
	with tf.name_scope('relu'):
		w_shape = (int(X.get_shape()[1]), 1)
		w = tf.Variable(tf.random_normal(w_shape), name='weights')
		b = tf.Variable(0.0, name='bias')
		z = tf.add(tf.matmul(X, w), b, name='z')
		return tf.maximum(z, 0, name='relu')
	
N_FEATURES = 3

if __name__ == '__main__':
	
	# Setting for Tensorboard
	now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
	root_logdir = 'tf_logs'
	logdir= './%s/relu-%s' % (root_logdir, now)
	
	
	X = tf.placeholder(tf.float32, shape=(None, N_FEATURES), name='X')

	relus = [relu(X) for i in range(5)]
	output = tf.add_n(relus, name='output')
	
	# For Tensorboard
	file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
	