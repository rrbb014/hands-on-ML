# make Computational Graph in Tensorflow p.299 - 301
import tensorflow as tf

PLAY_NUM = 3

x = tf.Variable(3, name='x')
y=  tf.Variable(4, name='y')
f = x*x*y + y + 2


# 01. session run
if PLAY_NUM == 1:
	sess = tf.Session()
	sess.run(x.initializer)
	sess.run(y.initializer)
	result = sess.run(f)
	print(result)
	sess.close()

# 02. Session with "With"	
elif PLAY_NUM == 2:
	with tf.Session() as sess:
		x.initializer.run()   # tf.get_default_session().run(x.initializer)를 호출하는 것과 동일
		y.initializer.run()
		result = f.eval()     # tf.get_default_session().run(f) 와 동일
		print(result)

# 03. global initializer
elif PLAY_NUM == 3:
	init = tf.global_variables_initializer()    # init 노드 
	with tf.Session() as sess:
		init.run()    # 모든 변수 초기화
		result = f.eval()
		print(result)

# 04. InteractiveSession for Jupyter or python shell
elif PLAY_NUM == 4:
	init = tf.global_variables_initializer()
	sess = tf.InteractiveSession()
	init.run()
	result = f.eval()
	print(result)
	sess.close()     # 인터렉티브세션을 사용하면 수동으로 닫아줘야한다.