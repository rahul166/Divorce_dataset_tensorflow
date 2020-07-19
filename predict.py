import tensorflow as tf
import pandas as pd
tf.compat.v1.disable_eager_execution()
imported=tf.compat.v1.train.import_meta_graph('model.ckpt.meta')
saver=tf.compat.v1.train.Saver()
data=pd.read_csv('divorce.csv')

test=data.iloc[:,:54]
test=test.to_numpy()

with tf.compat.v1.Session() as sess:
	# prediction = neural_network(x)
	imported.restore(sess, 'model.ckpt')
	saver.restore(sess,'model.ckpt')
	sess.run(tf.compat.v1.global_variables_initializer())
	prediction=	sess.run(['prediction:0'])	
	print(sess.run(prediction.eval(feed_dict={features:test[0]})))
















