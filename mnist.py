import tensorflow as tf
import binary_layer as binary
from tensorflow.examples.tutorials.mnist import input_data

def fully_connect_bn(pre_layer, output_dim, act, use_bias, training):
    pre_act = binary.dense_binary(pre_layer, output_dim, use_bias=use_bias)
    bn = binary.batch_normalization(pre_act, momentum=0.9, epsilon=1e-4, training=training)
    #bn = tf.layers.batch_normalization(pre_act, momentum=0.99, epsilon=0.00001, training=training)
    if act == None:
        output = bn
    else:
        output = act(bn)
    return output

x = tf.placeholder(tf.float32, shape=[None, 784])
target = tf.placeholder(tf.float32, shape=[None, 10])
training = tf.placeholder(tf.bool)

mnist = input_data.read_data_sets('./MNIST_data/', one_hot=True)
# convert class vectors to binary class vectors
for i in range(mnist.train.labels.shape[0]):
    mnist.train.labels[i] = mnist.train.labels[i] * 2 - 1 # -1 or 1 for hinge loss
for i in range(mnist.test.labels.shape[0]):
    mnist.test.labels[i] = mnist.test.labels[i] * 2 - 1
print(mnist.test.labels.shape)
print(mnist.test.images.shape)

layer0 = tf.layers.dropout(x, rate=0.2)

layer1 = fully_connect_bn(layer0, 4096, act=binary.binary_tanh_unit, use_bias=False, training=training)
layer1_dp = tf.layers.dropout(layer1, rate=0.5)

layer2 = fully_connect_bn(layer1_dp, 4096, act=binary.binary_tanh_unit, use_bias=False, training=training)
layer2_dp = tf.layers.dropout(layer2, rate=0.5)

layer3 = fully_connect_bn(layer2_dp, 4096, act=binary.binary_tanh_unit, use_bias=False, training=training)
layer3_dp = tf.layers.dropout(layer3, rate=0.5)

layer4 = fully_connect_bn(layer3_dp, 10, act=None, use_bias=False, training=training)

#out_act_training = tf.nn.softmax_cross_entropy_with_logits(logits=layer4, labels=target)
#out_act_testing = tf.nn.softmax(logits=layer4)
#out_act = tf.cond(training, lambda: out_act_training, lambda: out_act_testing)

loss = tf.keras.metrics.squared_hinge(target, layer4)

lr_start = 0.003
lr_end = 0.0000003
lr = tf.Variable(lr_start, name="lr")


opt = tf.train.AdamOptimizer(lr)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):   # when training, the moving_mean and moving_variance in the BN need to be updated.
    train_op = opt.apply_gradients(binary.compute_grads(loss, opt))

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(layer4, 1), tf.argmax(target, 1)), tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs = 1000
for i in range(epochs):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_op, feed_dict={x: batch_xs, target: batch_ys, training: True})

    print("acc %g, lr %g" % (sess.run(
                              accuracy, feed_dict={
                                  x: mnist.test.images,
                                  target: mnist.test.labels,
                                  training: False
                              }),
                            sess.run(lr)))

    new_lr = sess.run(lr) * (lr_end / lr_start)**(1. / epochs)
    sess.run(lr.assign(new_lr))
