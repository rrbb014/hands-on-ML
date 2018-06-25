# Tensorflow High level API for training DNN using MNIST 

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# 위의 데이터셋을 불러오면 (60000, 28, 28) 의 shape를 가져 이를 flatten함
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

# Validation Set 
X_train, X_valid = train_test_split(X_train, random_state=42)
y_train, y_valid = train_test_split(y_train, random_state=42)

# About book 
feature_cols = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[300, 100], n_classes=10, feature_columns=feature_cols)
dnn_clf = tf.contrib.learn.SKCompat(dnn_clf)
dnn_clf.fit(X_train, y_train, batch_size=50, steps=40000)

y_pred = dnn_clf.predict(X_valid)

# y_pred -> dict ('classes': [...], 'logits': [...], 'probabilities': [...])
print(accuracy_score(y_valid, y_pred['classes']))
