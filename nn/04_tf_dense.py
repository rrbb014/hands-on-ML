# tf.layer.dense usage
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import train_test_split

N_INPUTS = 28*28 # MNIST
N_HIDDEN1 = 300
N_HIDDEN2 = 100
N_OUTPUTS = 10
LEARNING_RATE = 0.01
N_EPOCHS = 10001
BATCH_SIZE = 50


def log_dir(prefix=''):
	"""Get Tensorboard log dir function"""
	now = datetime.now().strftime('%Y%m%d_%H%M%S')
	root_log_dir = './tf_logs'
	if prefix:
		prefix += '-'
	name = prefix + 'run-' + now
	return '{}/{}'.format(root_log_dir, name)
		
def shuffle_batch(X, y, batch_size):
	rnd_idx = np.random.permutation(len(X))
	n_batches = len(X) // batch_size
	for batch_idx in np.array_split(rnd_idx, n_batches):
		X_batch, y_batch = X[batch_idx], y[batch_idx]
		yield X_batch, y_batch

if __name__ == '__main__':
	# TensorBoard Setting
	logdir = log_dir('dense')
	
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
	
	with tf.name_scope('forward'):
		hidden1 = tf.layers.dense(X, N_HIDDEN1, name='hidden1', activation=tf.nn.relu)
		hidden2 = tf.layers.dense(hidden1, N_HIDDEN2, name='hidden2', activation=tf.nn.relu)
		logits = tf.layers.dense(hidden2, N_OUTPUTS, name='outputs')
		y_proba = tf.nn.softmax(logits)
	
	with tf.name_scope('loss'):
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
		loss = tf.reduce_mean(cross_entropy, name='loss')
		loss_summary = tf.summary.scalar('loss', loss)
		
	with tf.name_scope('train'):
		optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
		training_op = optimizer.minimize(loss)
	
	with tf.name_scope('eval'):
		correct = tf.nn.in_top_k(logits, y, 1)
		accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
		accuracy_summary = tf.summary.scalar('accuracy', accuracy)
	
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()
	
	file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
	
	m, n = X_train.shape
	n_batches = int(np.ceil(m / BATCH_SIZE))
	
	checkpoint_path = './model/my_deep_mnist_model.ckpt'
	checkpoint_epoch_path = checkpoint_path + '.epoch'
	final_model_path = './model/my_deep_mnist_model'

	best_loss = np.infty
	epochs_without_progress = 0
	max_epochs_without_progress = 50
	
	with tf.Session() as sess:
		if os.path.isfile(checkpoint_epoch_path):
			# 체크포인트 파일이 있으면 모델복원, epoch 숫자를 로드
			with open(checkpoint_epoch_path, 'rb') as f:
				start_epoch = int(f.read())
			print('이전 훈련이 중지되었습니다. epoch {}에서 시작합니다.'.format(start_epoch))
			saver.restore(sess, checkpoint_path)
		else:
			start_epoch = 0
			sess.run(init)
		
		for epoch in range(start_epoch, N_EPOCHS):
			for X_batch, y_batch in shuffle_batch(X_train, y_train, BATCH_SIZE):
				sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
			accuracy_val, loss_val, accuracy_summary_str, loss_summary_str = sess.run(
				[accuracy, loss, accuracy_summary, loss_summary],
				feed_dict={X:X_valid, y:y_valid})
			file_writer.add_summary(accuracy_summary_str, epoch)
			file_writer.add_summary(loss_summary_str, epoch)
			
			if epoch % 5 == 0:
				print('Epoch : ', epoch,
					'\t validation set ACC: {:.3f}%'.format(accuracy_val * 100),
					'\t loss {:.5f}'.format(loss_val))
				saver.save(sess, checkpoint_path)
				with open(checkpoint_epoch_path, 'wb') as f:
					f.write(b'%d' % epoch+1)
				if loss_val < best_loss:
					saver.save(sess, final_model_path)
					best_loss = loss_val
					
				else:
					epochs_without_progress += 5
					if epochs_without_progress > max_epochs_without_progress:
						print('Early Stopping')
						break