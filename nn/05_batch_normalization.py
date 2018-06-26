# MNIST DNN classifier using Batch Normalization

import numpy as np
import tensorflow as tf


def load_mnist_dataset():
    mnist = tf.contrib.learn.datasets.load_dataset('mnist')
    X_train = mnist.train.images
    y_train = mnist.train.labels.astype(np.int32)
    X_test = mnist.test.images
    y_test = mnist.test.labels.astype(np.int32)
    return X_train, y_train, X_test, y_test


class DNNClassifier:
	def __init__(self):
		self.conv1 = tf.layer
	
	def forward(self, X):
		
	
if __name__ == "__main__":
	X_train, y_train, X_test, y_test = load_mnist_dataset()
	
	