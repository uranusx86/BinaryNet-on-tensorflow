# coding=UTF-8
import tensorflow as tf
from tensorflow.python.framework import tensor_shape, ops
from tensorflow.python.ops import standard_ops, nn
import numpy as np

# Warning: if you have a @property getter/setter function in a class, must inherit from object class

all_layers = []

def hard_sigmoid(x):
    return tf.clip_by_value((x+1.)/2., 0, 1)

# The neurons' activations binarization function
# It behaves like the sign function during forward propagation
# And like:
#   hard_tanh(x) = 2*hard_sigmoid(x)-1
# during back propagation
def binary_tanh_unit(x):
    return 2.*hard_sigmoid(x)-1.

def binary_sigmoid_unit(x):
    return hard_sigmoid(x)

# The weights' binarization function,
# taken directly from the BinaryConnect github repository
# (which was made available by his authors)
def binarization(W, H, binary=True, deterministic=False, stochastic=False, srng=None):
    dim = W.get_shape().as_list()

    # (deterministic == True) <-> test-time <-> inference-time
    if not binary or (deterministic and stochastic):
        # print("not binary")
        Wb = W

    else:
        # [-1,1] -> [0,1]
        Wb = hard_sigmoid(W/H)
        # Wb = T.clip(W/H,-1,1)

        # Stochastic BinaryConnect
        '''
        if stochastic:
            # print("stoch")
            Wb = tf.cast(srng.binomial(n=1, p=Wb, size=tf.shape(Wb)), tf.float32)
        '''

        # Deterministic BinaryConnect (round to nearest)
        #else:
        # print("det")
        Wb = tf.round(Wb)

        # 0 or 1 -> -1 or 1
        Wb = tf.where(tf.equal(Wb,tf.constant(1.0)), tf.constant(H, shape=dim), tf.constant(-H, shape=dim))

    return Wb


class Dense_BinaryLayer(tf.layers.Dense):
    def __init__(self, output_dim,
               activation = None,
               use_bias = True,
               binary = True, stochastic = True, H=1., W_LR_scale="Glorot",
               kernel_initializer = tf.glorot_normal_initializer(),
               bias_initializer = tf.zeros_initializer(),
               kernel_regularizer = None,
               bias_regularizer = None,
               activity_regularizer = None,
               kernel_constraint = None,
               bias_constraint = None,
               trainable = True,
               name = None,
               **kwargs):
        super(Dense_BinaryLayer, self).__init__(units = output_dim,
               activation = activation,
               use_bias = use_bias,
               kernel_initializer = kernel_initializer,
               bias_initializer = bias_initializer,
               kernel_regularizer = kernel_regularizer,
               bias_regularizer = bias_regularizer,
               activity_regularizer = activity_regularizer,
               kernel_constraint = kernel_constraint,
               bias_constraint = bias_constraint,
               trainable = trainable,
               name = name,
               **kwargs)

        self.binary = binary
        self.stochastic = stochastic

        self.H = H
        self.W_LR_scale = W_LR_scale

        all_layers.append(self)

    def build(self, input_shape):
        num_inputs = tensor_shape.TensorShape(input_shape).as_list()[-1]
        num_units = self.units
        print(num_inputs)
        print(num_units)

        self.H = np.float32(np.sqrt(1.5 / (num_inputs + num_units)))   # weight init method
        self.W_LR_scale = np.float32(1. / self.H)                      # each layer learning rate

        self.kernel_initializer = tf.random_uniform_initializer(-self.H, self.H)

        super(Dense_BinaryLayer, self).build(input_shape)
        self.built = False

        self.b_kernel = self.add_variable('binary_weight',
                                    shape=[input_shape[-1], self.units],
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint,
                                    dtype=self.dtype,
                                    trainable=False)
        self.built = True

        tf.add_to_collection('real_weight', self.kernel)
        tf.add_to_collection('binary_weight', self.b_kernel)


    def call(self, inputs):
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        shape = inputs.get_shape().as_list()

        # binarization weight
        self.b_kernel = binarization(self.kernel, self.H)
        r_kernel = self.kernel
        self.kernel = self.b_kernel

        if len(shape) > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(inputs, self.kernel, [[len(shape) - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if context.in_graph_mode():
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            outputs = standard_ops.matmul(inputs, self.kernel)

        # restore weight
        self.kernel = r_kernel

        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

# Functional interface for the Dense_BinaryLayer class.
def dense_binary(
    inputs, units,
    activation=None,
    use_bias=True,
    kernel_initializer=tf.glorot_normal_initializer(),
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    trainable=True,
    name=None,
    reuse=None):

    layer = Dense_BinaryLayer(units,
                activation=activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint,
                bias_constraint=bias_constraint,
                trainable=trainable,
                name=name,
                dtype=inputs.dtype.base_dtype,
                _scope=name,
                _reuse=reuse)
    return layer.apply(inputs)


class Conv2D_BinaryLayer(tf.layers.Conv2D):
    '''
    __init__(): init variable
    conv2d():   Functional interface for the 2D convolution layer.
                This layer creates a convolution kernel that is convolved(actually cross-correlated)
                with the layer input to produce a tensor of outputs.
    apply():    Apply the layer on a input, This simply wraps `self.__call__`
    __call__(): Wraps `call` and will be call build(), applying pre- and post-processing steps
    call():     The logic of the layer lives here
    '''

    def __init__(self, kernel_num,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               activation=None,
               use_bias=True,
               data_format='channels_last',
               dilation_rate=(1, 1),
               kernel_initializer=None,
               bias_initializer=tf.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
        super(Conv2D_BinaryLayer, self).__init__(filters = kernel_num,
               kernel_size = kernel_size,
               strides = strides,
               padding = padding,
               data_format = data_format,
               dilation_rate = dilation_rate,
               activation = activation,
               use_bias = use_bias,
               kernel_initializer = kernel_initializer,
               bias_initializer = bias_initializer,
               kernel_regularizer = kernel_regularizer,
               bias_regularizer = bias_regularizer,
               activity_regularizer = activity_regularizer,
               kernel_constraint = kernel_constraint,
               bias_constraint = bias_constraint,
               trainable = trainable,
               name = name,
               **kwargs)

        self.b_kernel = tf.Variable(self.kernel, name='binary_weight')

        tf.add_to_collection('real_weight', self.kernel)
        tf.add_to_collection('binary_weight', self.b_kernel)

        all_layers.append(self)

    @property
    def b_kernel():
        return self.b_kernel

    def call(self, inputs):
        # binarization weight
        self.b_kernel = binarization(self.kernel)
        r_kernel = self.kernel
        self.kernel = self.b_kernel

        outputs = self._convolution_op(inputs, self.kernel)

        # restore weight
        self.kernel = r_kernel

        if self.use_bias:
            if self.data_format == 'channels_first':
                if self.rank == 1:
                    # nn.bias_add does not accept a 1D input tensor.
                    bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                    outputs += bias
                if self.rank == 2:
                    outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
                if self.rank == 3:
                    # As of Mar 2017, direct addition is significantly slower than
                    # bias_add when computing gradients. To use bias_add, we collapse Z
                    # and Y into a single dimension to obtain a 4D input tensor.
                    outputs_shape = outputs.shape.as_list()
                    outputs_4d = array_ops.reshape(outputs,
                                                 [outputs_shape[0], outputs_shape[1],
                                                  outputs_shape[2] * outputs_shape[3],
                                                  outputs_shape[4]])
                    outputs_4d = nn.bias_add(outputs_4d, self.bias, data_format='NCHW')
                    outputs = array_ops.reshape(outputs_4d, outputs_shape)
            else:
                outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

# Functional interface for the Conv2D_BinaryLayer.
def conv2d_binary(inputs,
           filters,
           kernel_size,
           strides = (1, 1),
           padding = 'valid',
           data_format = 'channels_last',
           dilation_rate = (1, 1),
           activation = None,
           use_bias = True,
           kernel_initializer = None,
           bias_initializer = tf.zeros_initializer(),
           kernel_regularizer = None,
           bias_regularizer = None,
           activity_regularizer = None,
           kernel_constraint = None,
           bias_constraint = None,
           trainable = True,
           name = None,
           reuse = None):

    layer = Conv2D_BinaryLayer(
                filters = filters,
                kernel_size = kernel_size,
                strides = strides,
                padding = padding,
                data_format = data_format,
                dilation_rate = dilation_rate,
                activation = activation,
                use_bias = use_bias,
                kernel_initializer = kernel_initializer,
                bias_initializer = bias_initializer,
                kernel_regularizer = kernel_regularizer,
                bias_regularizer = bias_regularizer,
                activity_regularizer = activity_regularizer,
                kernel_constraint = kernel_constraint,
                bias_constraint = bias_constraint,
                trainable = trainable,
                name = name,
                dtype = inputs.dtype.base_dtype,
                _reuse = reuse,
                _scope = name)
    return layer.apply(inputs)

# This function computes the gradient of the binary weights
#
# use binary weight compute gradients
# use optimizer to get float value updated weight value by given gradient
# clipping float value updated weight, and apply
def compute_grads(loss, opt):
    gradients_list  = []
    parameters_list = []

    for layer in all_layers[::-1]:
        # binary weight and bias
        params = []
        '''
        for var in layer.trainable_variables:
            if not var.name.endswith('kernel:0'):
                params.append(var)
            else:
                params.append(layer.b_kernel)
        '''
        params=layer.b_kernel

        #if len(params) > 0:
        #compute gradients for params
        grad = tf.gradients(loss, params)  # if len(params)=n, return [grad[0],...,grad[n-1]]

        #process gradients
        clipped_gradients = clipping_scaling(params, grad[0], layer.W_LR_scale, layer.H)

        gradients_list.append(clipped_gradients)
        #parameters_list.append(layer.trainable_variables) # update real value weight
        parameters_list.append(layer.kernel)

    print(gradients_list)
    print(parameters_list)
    return zip(gradients_list, parameters_list)


# This functions clips the weights after the parameter update
def clipping_scaling(gradient, update_gradients, W_LR_scale, H):
    print("W_LR_scale = "+str(W_LR_scale))
    print("H = "+str(H))
    #W_LR_scale為learning_rate擴大倍數，updates[param]為更新後的權重，param為未更新權重
    #grad_scale = gradient + W_LR_scale*(tf.subtract(update_gradients, gradient))
    grad_clip = tf.clip_by_value(update_gradients, -H, H)

    return grad_clip