import pandas as pd
import tensorflow as tf
import numpy as np
import pickle


tf.compat.v1.disable_eager_execution()
def neural_network(data):
	# print(data)
	hidden_layer1={'weights':tf.Variable(tf.compat.v1.random_normal([54,500])),
					'bias':tf.Variable(tf.compat.v1.random_normal([500]))
					}

	hidden_layer2={'weights':tf.Variable(tf.compat.v1.random_normal([500,500])),
					'bias':tf.Variable(tf.compat.v1.random_normal([500]))
					}


	out={'weights':tf.Variable(tf.compat.v1.random_normal([500,2])),
					'bias':tf.Variable(tf.compat.v1.random_normal([2]))
					}


	layer1=tf.add(tf.matmul(data,hidden_layer1['weights']),hidden_layer1['bias'])
	layer1=tf.nn.relu(layer1)
	layer2=tf.add(tf.matmul(layer1,hidden_layer2['weights']),hidden_layer2['bias'])
	layer2=tf.nn.relu(layer2)
	out_layer=tf.add(tf.matmul(layer2,out['weights']),out['bias'])

	return out_layer


with open('data','rb') as f:
	data=pickle.load(f)
# print(len(data[0]),len(data[0][0]))
features=tf.compat.v1.placeholder(tf.float32,[None,54])
labels=tf.compat.v1.placeholder(tf.float32,[None,2])
prediction=neural_network(features)
cost=tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=labels))
optimizer=tf.compat.v1.train.AdamOptimizer().minimize(cost)

saver=tf.compat.v1.train.Saver()
with tf.compat.v1.Session() as sess:
	sess.run(tf.compat.v1.global_variables_initializer())

	hm_epochs=100
	for i in range(hm_epochs):
		epoch_cost=0;
		_,c=sess.run([optimizer,cost],feed_dict={features:data[0],labels:data[1]})
		print("epoch",i,"completed out of",hm_epochs,"cost:",c)


	# saver.save(sess,'model.ckpt')
	# print("Model saved")
	correct=tf.math.equal(tf.argmax(prediction,1),tf.argmax(labels,1))

	accuracy=tf.math.reduce_mean(tf.cast(correct,float))
	print("accuracy:",accuracy.eval(feed_dict={features:data[0],labels:data[1]}))
	# print(neural_network(data[0][0]))
	# test=np.array(data[0][0])
	# # test=np.transpose(test)
	# print(test.reshape(1,-1))
	# test=test.reshape(1,-1)
	# test=list(test)
	# # print(np.transpose(test).shape)
	# print(sess.run(prediction.eval(feed_dict={features:data[0][0].reshape(1,-1)})))

















