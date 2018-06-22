# Sharing Variable between several graph

import numpy as np
import tensorflow as tf

from datetime import datetime

N_FEATURES = 5
THRESHOLD = 0.0

def relu(X):
	with tf.variable_scope('relu', reuse=True):
		threshold = tf.get_variable('threshold')
		w = tf.Variable(tf.random_normal((N_FEATURES, 1)), name='weight')
		b = tf.Variable(1.0, name='bias')
		z = tf.add(tf.matmul(X, w), b, name='z')
		return tf.maximum(z, threshold, name='relu_max')

		
if __name__ == '__main__':
	# Tensorboard Setting
	now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
	root_logdir = 'tf_logs'
	logdir = './%s/share_relu-%s' % (root_logdir, now)
	
	with tf.variable_scope('relu'):
		threshold = tf.get_variable('threshold', shape=(),
									initializer=tf.constant_initializer(THRESHOLD))
		
	X = tf.placeholder(tf.float32, shape=(None, N_FEATURES), name='X')
	relus = [relu(X) for _ in range(5)]
	output = tf.add_n(relus, name='output')
	
	# Tensorboard
	file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
	