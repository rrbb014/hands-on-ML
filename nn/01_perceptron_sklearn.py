import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data[:, (2,3)]                # 꽃잎의 길이, 너비
y = (iris.target == 0).astype(np.int)  # Setosa 인가?

per_clf = Perceptron(random_state=42)
per_clf.fit(X, y)

y_pred = per_clf.predict([[2, 0.5]]) 
print(y_pred)