import tensorflow as tf
import os
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# load dataset MNIST
mnist = input_data.read_data_sets("./data/", one_hot=True)


x = tf.placeholder(dtype=tf.float32, shape=(None, 784))
y = tf.placeholder(dtype=tf.float32, shape=(None, 10))

W = tf.get_variable(name='weight', shape=(784,10))
b = tf.get_variable(name='bias', shape=10)

layermul = tf.matmul(x, W)
layer = tf.add(layermul, b)
layer_sigmoid = tf.nn.softmax(layer)

loss = tf.losses.mean_squared_error(y, layer_sigmoid)
loss_op = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=y, logits=layer_sigmoid))
tf.summary.scalar("loss", loss_op)

correct_prediction = tf.equal(tf.argmax(layer_sigmoid, 1), tf.argmax(y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, "float"))
tf.summary.scalar("training_accuracy", accuracy_op)

train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_op)

init = tf.global_variables_initializer()

summary_op = tf.summary.merge_all()
saver = tf.train.Saver()

iter = 0
batch_size = 100

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('./checkpoints/')
    if ckpt:
        saver.restore(sess, './checkpoints/-38000')
        print("Successfully restored model")
    else:
        sess.run(init)

    writer = tf.summary.FileWriter('./graphs', sess.graph)

    for iter in range(40000):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, loss, tr_acc = sess.run([train_op, loss_op, accuracy_op], feed_dict={x: batch_x, y: batch_y})

        if iter % 50 == 0:
            print("Iter %d: loss %f, train_accuracy %f" % (iter, loss, tr_acc))
            summary = sess.run(summary_op, feed_dict={x: batch_x, y: batch_y})
            writer.add_summary(summary, iter)

            if iter % 2000 == 0:
                saver.save(sess, './checkpoints/', global_step=iter)
