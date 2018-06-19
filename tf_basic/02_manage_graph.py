# Managing TF graph p.301
# 여러 계산 그래프를 만들때
# 만약, Jupyter 나 python shell 에서 실행할땐 default graph에 여러 노드가 중복 실행되는 경우가 있으니,
# tf.reset_default_graph() 를 실행해주면 초기화되서 유용하다

import tensorflow as tf

x1 = tf.Variable(1)
print("'X1's graph is tf's default graph? : ", x1.graph is tf.get_default_graph())

# Make other graph
graph = tf.Graph()
# must add explicitly "with graph.as_default()"
with graph.as_default():
	x2 = tf.Variable(2)
	print("'X2's graph is tf's default graph? : ", x2.graph is tf.get_default_graph())

print()
print("==== After graph's computation ended ====")
print()
print("'X1's graph is tf's default graph? : ", x1.graph is tf.get_default_graph())
print("'X2's graph is tf's default graph? : ", x2.graph is tf.get_default_graph())