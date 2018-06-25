# Tensorflow low level API for training MLP

import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import train_test_split

N_INPUTS = 28*28 # MNIST
N_HIDDEN1 = 300
N_HIDDEN2 = 100
N_OUTPUTS = 10
LEARNING_RATE = 0.01
N_EPOCHS = 40
BATCH_SIZE = 50


def neuron_layer(X, n_neurons, name, activation=None):
	with tf.name_scope(name):
		n_inputs = int(X.get_shape()[1])
		stddev = 2 / np.sqrt(N_INPUTS)

		# 웨이트의 초기화를 2 / sqrt(n_inputs + n_neurons) 의 truncated_normal dist로 랜덤 초기화
		# Xavier initialization
		init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
		W = tf.Variable(init, name='kernel')
		b = tf.Variable(tf.zeros([n_neurons]), name='bias')
		z = tf.matmul(X, W) + b
		if activation is not None:
			return activation(z)
		else:
			return z

def shuffle_batch(X, y, batch_size):
	rnd_idx = np.random.permutation(len(X))
	n_batches = len(X) // batch_size
	for batch_idx in np.array_split(rnd_idx, n_batches):
		X_batch, y_batch = X[batch_idx], y[batch_idx]
		yield X_batch, y_batch
		
if __name__ == "__main__":
	# TensorBoard Setting
	now = datetime.now().strftime('%Y%m%d_%H%M%S')
	root_log_dir = 'tf_logs'
	logdir = './{}/run-{}'.format(root_log_dir, now)
	
	# Prepare MNIST Dataset
	(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
	X_train = X_train.astype(np.float32).reshape(-1, N_INPUTS) / 255.0
	X_test = X_test.astype(np.float32).reshape(-1, N_INPUTS) / 255.0
	y_train = y_train.astype(np.int32)
	y_test = y_test.astype(np.int32)
	
	# Validation Set 
	X_train, X_valid = train_test_split(X_train, random_state=42)
	y_train, y_valid = train_test_split(y_train, random_state=42)

	tf.reset_default_graph()
	
	X = tf.placeholder(tf.float32, shape=(None, N_INPUTS), name='X')
	y = tf.placeholder(tf.int32, shape=(None), name='y')

	with tf.name_scope('dnn'):
		hidden1 = neuron_layer(X, N_HIDDEN1, name='hidden1', activation=tf.nn.relu)
		hidden2 = neuron_layer(hidden1, N_HIDDEN2, name='hidden2', activation=tf.nn.relu)
		logits = neuron_layer(hidden2, N_OUTPUTS, name='outputs')
	
	with tf.name_scope('loss'):
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
		loss = tf.reduce_mean(cross_entropy, name='loss')
	
	with tf.name_scope('train'):
		optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
		training_op = optimizer.minimize(loss)

	with tf.name_scope('eval'):
		correct = tf.nn.in_top_k(logits, y, 1)
		accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
	
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()
	
	# Tensorboard Setting
	acc_summary = tf.summary.scalar('ACCURACY', accuracy)
	file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
	
	with tf.Session() as sess:
		sess.run(init)
		cnt = 1
		for epoch in range(N_EPOCHS):
			for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size=BATCH_SIZE):
				sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
			acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
			acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
			
			# TensorBoard
			summary_str = acc_summary.eval(feed_dict={X: X_valid, y: y_valid})
			file_writer.add_summary(summary_str, epoch+cnt)
			print(epoch, "Batch data Accuracy: ", acc_batch, " Validation set Accuracy: ", acc_valid)
			cnt += 1
		save_path = saver.save(sess, './model/my_model_final.ckpt')
	
	file_writer.close()
	
	with tf.Session() as sess:
		saver.restore(sess, './model/my_model_final.ckpt')
		X_new_scaled = X_test[:20]
		Z = logits.eval(feed_dict={X: X_new_scaled})
		y_pred = np.argmax(Z, axis=1)
	
	print('predicted class: ', y_pred)
	print('real class : ', y_test[:20])
	