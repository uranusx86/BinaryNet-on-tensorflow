import tensorflow as tf
import binary_layer as binary
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from mnist import download_mnist

def fully_connect_bn(pre_layer, output_dim, act, use_bias, training):
    pre_act = binary.dense_binary(pre_layer, output_dim,
                                    use_bias = use_bias,
                                    activation = None,
                                    kernel_constraint = lambda w: tf.clip_by_value(w, -1.0, 1.0))
    bn = binary.batch_normalization(pre_act, momentum=0.9, epsilon=1e-4, training=training)
    if act == None:
        output = bn
    else:
        output = act(bn)
    return output

def no_scale_dropout(pre_layer, drop_rate, training):
    drop_layer = tf.layers.dropout(pre_layer, rate=drop_rate, training=training)
    #return tf.cond(training, lambda: drop_layer*(1-drop_rate), lambda: drop_layer)
    return drop_layer

# A function which shuffles a dataset
def shuffle(X,y):
    print(len(X))
    shuffle_parts = 1
    chunk_size = int(len(X)/shuffle_parts)
    shuffled_range = np.arange(chunk_size)

    X_buffer = np.copy(X[0:chunk_size])
    y_buffer = np.copy(y[0:chunk_size])

    for k in range(shuffle_parts):

        np.random.shuffle(shuffled_range)

        for i in range(chunk_size):

            X_buffer[i] = X[k*chunk_size+shuffled_range[i]]
            y_buffer[i] = y[k*chunk_size+shuffled_range[i]]

        X[k*chunk_size:(k+1)*chunk_size] = X_buffer
        y[k*chunk_size:(k+1)*chunk_size] = y_buffer

    return X,y

# This function trains the model a full epoch (on the whole dataset)
def train_epoch(X, y, sess, batch_size=100):
    batches = int(len(X)/batch_size)
    for i in range(batches):
        sess.run([train_kernel_op, train_other_op],
            feed_dict={ x: X[i*batch_size:(i+1)*batch_size],
                        target: y[i*batch_size:(i+1)*batch_size],
                        training: True})

x = tf.placeholder(tf.float32, shape=[None, 784])
target = tf.placeholder(tf.float32, shape=[None, 10])
training = tf.placeholder(tf.bool)

download_mnist.maybe_download('./mnist/MNIST_data/')
mnist = input_data.read_data_sets('./mnist/MNIST_data/', one_hot=True)

# convert class vectors to binary class vectors
for i in range(mnist.train.images.shape[0]):
    mnist.train.images[i] = mnist.train.images[i] * 2 - 1
for i in range(mnist.test.images.shape[0]):
    mnist.test.images[i] = mnist.test.images[i] * 2 - 1
for i in range(mnist.train.labels.shape[0]):
    mnist.train.labels[i] = mnist.train.labels[i] * 2 - 1 # -1 or 1 for hinge loss
for i in range(mnist.test.labels.shape[0]):
    mnist.test.labels[i] = mnist.test.labels[i] * 2 - 1
print(mnist.test.labels.shape)
print(mnist.test.images.shape)

layer0 = no_scale_dropout(x, drop_rate=0.2, training=training)

layer1 = fully_connect_bn(layer0, 4096, act=binary.binary_tanh_unit, use_bias=True, training=training)
layer1_dp = no_scale_dropout(layer1, drop_rate=0.5, training=training)

layer2 = fully_connect_bn(layer1_dp, 4096, act=binary.binary_tanh_unit, use_bias=True, training=training)
layer2_dp = no_scale_dropout(layer2, drop_rate=0.5, training=training)

layer3 = fully_connect_bn(layer2_dp, 4096, act=binary.binary_tanh_unit, use_bias=True, training=training)
layer3_dp = no_scale_dropout(layer3, drop_rate=0.5, training=training)

layer4 = fully_connect_bn(layer3_dp, 10, act=None, use_bias=True, training=training)

#out_act_training = tf.nn.softmax_cross_entropy_with_logits(logits=layer4, labels=target)
#out_act_testing = tf.nn.softmax(logits=layer4)
#out_act = tf.cond(training, lambda: out_act_training, lambda: out_act_testing)

loss = tf.reduce_mean(tf.square(tf.maximum(0.,1.-target*layer4)))

epochs = 1000
lr_start = 0.003
lr_end = 0.0000003
lr_decay = (lr_end / lr_start)**(1. / epochs)
global_step1 = tf.Variable(0, trainable=False)
global_step2 = tf.Variable(0, trainable=False)
lr1 = tf.train.exponential_decay(lr_start, global_step=global_step1, decay_steps=int(mnist.train.images.shape[0]/100), decay_rate=lr_decay)
lr2 = tf.train.exponential_decay(lr_start, global_step=global_step2, decay_steps=int(mnist.train.images.shape[0]/100), decay_rate=lr_decay)

sess = tf.Session()
saver = tf.train.Saver()
#saver.restore(sess, "model/model.ckpt")

other_var = [var for var in tf.trainable_variables() if not var.name.endswith('kernel:0')]
opt = binary.AdamOptimizer(binary.get_all_LR_scale(), lr1)
opt2 = tf.train.AdamOptimizer(lr2)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):   # when training, the moving_mean and moving_variance in the BN need to be updated.
    train_kernel_op = opt.apply_gradients(binary.compute_grads(loss, opt),  global_step=global_step1)
    train_other_op  = opt2.minimize(loss, var_list=other_var,  global_step=global_step2)


accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(layer4, 1), tf.argmax(target, 1)), tf.float32))
sess.run(tf.global_variables_initializer())

old_acc = 0.0
X_train, y_train = shuffle(mnist.train.images, mnist.train.labels)
for i in range(epochs):
    train_epoch(X_train, y_train, sess)
    X_train, y_train = shuffle(mnist.train.images, mnist.train.labels)

    hist = sess.run([accuracy, opt._lr],
                    feed_dict={
                        x: mnist.test.images,
                        target: mnist.test.labels,
                        training: False
                    })
    print(hist)

    if hist[0] > old_acc:
        old_acc = hist[0]
        save_path = saver.save(sess, "./mnist/model/model.ckpt")

