# Lifecycle of Node value  p.302
import tensorflow as tf

PLAY_NUM = 2

if __name__ == "__main__":
		
	w = tf.constant(3)
	x = w + 2
	y = x + 5
	z = x * 3
	if PLAY_NUM is 1:
		with tf.Session() as sess:
			print(y.eval()) # 10, x와 w를 평가하는데, w를 먼저 -> x -> y를 평가.
			print(z.eval()) # 15, x,w를 평가하는데, x와 w를 중복해서 다시 평가한다. 이전 평가는 활용하지않음.
	else:
		# w와 x가 중복 평가되는 손실을 막기위해, 한번의 그래프 실행으로 y, z를 모두 평가하게 한다.
		with tf.Session() as sess:
			y_val, z_val = sess.run([y, z])
			print(y_val)
			print(z_val)
	