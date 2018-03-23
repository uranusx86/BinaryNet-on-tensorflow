import tensorflow as tf
from cifar10 import cifar10_api as cifar10
import binary_layer
import numpy as np
# acc: 86.18%

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
            feed_dict={ input: X[i*batch_size:(i+1)*batch_size],
                        target: y[i*batch_size:(i+1)*batch_size],
                        training: True})

def conv_bn(pre_layer, kernel_num, kernel_size, padding, activation, training, epsilon=1e-4, alpha=.1, binary=True, stochastic=False, H=1., W_LR_scale="Glorot"):
    conv = binary_layer.conv2d_binary(pre_layer, kernel_num, kernel_size, padding=padding, binary=binary, stochastic=stochastic, H=H, W_LR_scale=W_LR_scale)
    bn = binary_layer.batch_normalization(conv, epsilon=epsilon, momentum = 1-alpha, training=training)
    output = activation(bn)
    return output

def conv_pool_bn(pre_layer, kernel_num, kernel_size, padding, pool_size, activation, training, epsilon=1e-4, alpha=.1, binary=True, stochastic=False, H=1., W_LR_scale="Glorot"):
    conv = binary_layer.conv2d_binary(pre_layer, kernel_num, kernel_size, padding=padding, binary=binary, stochastic=stochastic, H=H, W_LR_scale=W_LR_scale)
    pool = tf.layers.max_pooling2d(conv, pool_size=pool_size, strides=pool_size)
    bn = binary_layer.batch_normalization(pool, epsilon=epsilon, momentum = 1-alpha, training=training)
    output = activation(bn)
    return output

def fully_connect_bn(pre_layer, output_dim, act, use_bias, training, epsilon=1e-4, alpha=.1, binary=True, stochastic=False, H=1., W_LR_scale="Glorot"):
    pre_act = binary_layer.dense_binary(pre_layer, output_dim,
                                    use_bias = use_bias,
                                    kernel_constraint = lambda w: tf.clip_by_value(w, -1.0, 1.0))
    bn = binary_layer.batch_normalization(pre_act, momentum=1-alpha, epsilon=epsilon, training=training)
    if act == None:
        output = bn
    else:
        output = act(bn)
    return output

cifar10.maybe_download_and_extract()
images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()
print("Size of:")
print("- Training-set:\t\t{}".format(len(images_train)))
print("- Test-set:\t\t{}".format(len(images_test)))

# Inputs in the range [-1,+1]
images_train = images_train * 2. - 1.
images_test = images_test * 2. - 1.
# for hinge loss
labels_train = labels_train * 2. - 1.
labels_test = labels_test * 2. - 1.

# alpha is the exponential moving average factor
alpha = .1
print("alpha = "+str(alpha))
epsilon = 1e-4
print("epsilon = "+str(epsilon))

# BinaryOut
activation = binary_layer.binary_tanh_unit
print("activation = binary_net.binary_tanh_unit")

# BinaryConnect
binary = True
print("binary = "+str(binary))
stochastic = False
print("stochastic = "+str(stochastic))
# (-H,+H) are the two binary values
H = 1.
print("H = "+str(H))
W_LR_scale = "Glorot" # "Glorot" means we are using the coefficients from Glorot's paper
print("W_LR_scale = "+str(W_LR_scale))

# Training parameters
num_epochs = 500
print("num_epochs = "+str(num_epochs))

training = tf.placeholder(tf.bool)
input = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
target = tf.placeholder(tf.float32, shape=[None, 10])

######### Build CNN ###########
cnn = conv_bn(input, 128, (3,3), padding='same', activation=activation, training=training)
cnn = conv_pool_bn(cnn, 128, (3,3), padding='same', pool_size=(2,2), activation=activation, training=training)

cnn = conv_bn(cnn, 256, (3,3), padding='same', activation=activation, training=training)
cnn = conv_pool_bn(cnn, 256, (3,3), padding='same', pool_size=(2,2), activation=activation, training=training)

cnn = conv_bn(cnn, 512, (3,3), padding='same', activation=activation, training=training)
cnn = conv_pool_bn(cnn, 512, (3,3), padding='same', pool_size=(2,2), activation=activation, training=training)

cnn = tf.layers.flatten(cnn)

cnn = fully_connect_bn(cnn, 1024, act=activation, use_bias=True, training=training)
cnn = fully_connect_bn(cnn, 1024, act=activation, use_bias=True, training=training)
train_output = fully_connect_bn(cnn, 10, act=None, use_bias=True, training=training)

loss = tf.keras.metrics.squared_hinge(target, train_output)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(train_output, 1), tf.argmax(target, 1)), tf.float32))

train_batch_size = 50
lr_start = 0.001
lr_end = 0.0000003
lr_decay = (lr_end / lr_start)**(1. / num_epochs)
global_step1 = tf.Variable(0, trainable=False)
global_step2 = tf.Variable(0, trainable=False)
lr1 = tf.train.exponential_decay(lr_start, global_step=global_step1, decay_steps=int(len(images_train)/train_batch_size), decay_rate=lr_decay)
lr2 = tf.train.exponential_decay(lr_start, global_step=global_step2, decay_steps=int(len(images_train)/train_batch_size), decay_rate=lr_decay)

other_var = [var for var in tf.trainable_variables() if not var.name.endswith('kernel:0')]
opt = binary_layer.AdamOptimizer(binary_layer.get_all_LR_scale(), lr1)
opt2 = tf.train.AdamOptimizer(lr2)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):   # when training, the moving_mean and moving_variance in the BN need to be updated.
    train_kernel_op = opt.apply_gradients(binary_layer.compute_grads(loss, opt),  global_step=global_step1)
    train_other_op  = opt2.minimize(loss, var_list=other_var,  global_step=global_step2)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

print("batch size = ", train_batch_size)
old_acc = 0.0
for j in range(num_epochs):
    train_data, train_label = shuffle(images_train, labels_train)
    train_epoch(train_data, train_label, sess, train_batch_size)

    acc = 0.0
    for i in range(int(len(images_test)/train_batch_size)):
        acc += sess.run(accuracy,
                feed_dict={
                    input: images_test[i*train_batch_size:(i+1)*train_batch_size],
                    target: labels_test[i*train_batch_size:(i+1)*train_batch_size],
                    training: False
                })
    acc /= (len(images_test)/train_batch_size)

    print("acc: %g, lr: %g" % (acc, sess.run(opt._lr)))

    if acc > old_acc:
        old_acc = acc
        save_path = saver.save(sess, "./cifar10/model/model.ckpt")
