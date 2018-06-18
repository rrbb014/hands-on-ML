# Managing TF graph p.301
# 여러 계산 그래프를 만들때
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