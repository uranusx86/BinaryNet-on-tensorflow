# coding=UTF-8
import tensorflow as tf
from tensorflow.python.framework import tensor_shape, ops
from tensorflow.python.ops import standard_ops, nn
import numpy as np

# Warning: if you have a @property getter/setter function in a class, must inherit from object class

all_layers = []

def hard_sigmoid(x):
    return tf.clip_by_value((x+1.)/2., 0, 1)

def round_through(x):
    '''Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    a op that behave as f(x) in forward mode,
    but as g(x) in the backward mode.
    '''
    rounded = tf.round(x)
    return x + tf.stop_gradient(rounded - x)

# The neurons' activations binarization function
# It behaves like the sign function during forward propagation
# And like:
#   hard_tanh(x) = 2*hard_sigmoid(x)-1
# during back propagation
def binary_tanh_unit(x):
    return 2.*round_through(hard_sigmoid(x))-1.

def binary_sigmoid_unit(x):
    return round_through(hard_sigmoid(x))

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
        #Wb = hard_sigmoid(W/H)
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
        #Wb = round_through(Wb)

        # 0 or 1 -> -1 or 1
        #Wb = tf.where(tf.equal(Wb,tf.constant(1.0)), tf.constant(H, shape=dim), tf.constant(-H, shape=dim))
        #Wb = H * Wb
        Wb = H * binary_tanh_unit(W / H)

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
        self.kernel_constraint = lambda w: tf.clip_by_value(w, -self.H, self.H)

        self.b_kernel = 0  # add_variable must execute before call build()

        super(Dense_BinaryLayer, self).build(input_shape)

        tf.add_to_collection('real_weight', self.kernel)
        #tf.add_to_collection('binary_weight', self.b_kernel)


    def call(self, inputs):
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        shape = inputs.get_shape().as_list()

        # binarization weight
        self.b_kernel = binarization(self.kernel, self.H)
        #r_kernel = self.kernel
        #self.kernel = self.b_kernel

        if len(shape) > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(inputs, self.b_kernel, [[len(shape) - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if context.in_graph_mode():
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            outputs = standard_ops.matmul(inputs, self.b_kernel)

        # restore weight
        #self.kernel = r_kernel

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

# Not yet binarized
class BatchNormalization(tf.layers.BatchNormalization):
  def __init__(self,
               axis = -1,
               momentum = 0.99,
               epsilon = 1e-3,
               center = True,
               scale = True,
               beta_initializer = tf.zeros_initializer(),
               gamma_initializer = tf.ones_initializer(),
               moving_mean_initializer = tf.zeros_initializer(),
               moving_variance_initializer = tf.ones_initializer(),
               beta_regularizer = None,
               gamma_regularizer = None,
               beta_constraint = None,
               gamma_constraint = None,
               renorm = False,
               renorm_clipping = None,
               renorm_momentum = 0.99,
               fused = None,
               trainable = True,
               name = None,
               **kwargs):
      super(BatchNormalization, self).__init__(axis = axis,
                                     momentum = momentum,
                                     epsilon = epsilon,
                                     center = center,
                                     scale = scale,
                                     beta_initializer = beta_initializer,
                                     gamma_initializer = gamma_initializer,
                                     moving_mean_initializer = moving_mean_initializer,
                                     moving_variance_initializer = moving_variance_initializer,
                                     beta_regularizer = beta_regularizer,
                                     gamma_regularizer = gamma_regularizer,
                                     beta_constraint = beta_constraint,
                                     gamma_constraint = gamma_constraint,
                                     renorm = renorm,
                                     renorm_clipping = renorm_clipping,
                                     renorm_momentum = renorm_momentum,
                                     fused = fused,
                                     trainable = trainable,
                                     name = name,
                                     **kwargs)
      all_layers.append(self)

  def build(self, input_shape):
      super(BatchNormalization, self).build(input_shape)
      self.W_LR_scale = np.float32(1.)

# Functional interface for the batch normalization layer.
def batch_normalization(inputs,
                        axis=-1,
                        momentum=0.99,
                        epsilon=1e-3,
                        center=True,
                        scale=True,
                        beta_initializer=tf.zeros_initializer(),
                        gamma_initializer=tf.ones_initializer(),
                        moving_mean_initializer=tf.zeros_initializer(),
                        moving_variance_initializer=tf.ones_initializer(),
                        beta_regularizer=None,
                        gamma_regularizer=None,
                        beta_constraint=None,
                        gamma_constraint=None,
                        training=False,
                        trainable=True,
                        name=None,
                        reuse=None,
                        renorm=False,
                        renorm_clipping=None,
                        renorm_momentum=0.99,
                        fused=None):
    layer = BatchNormalization(axis = axis,
                              momentum = momentum,
                              epsilon = epsilon,
                              center = center,
                              scale = scale,
                              beta_initializer = beta_initializer,
                              gamma_initializer = gamma_initializer,
                              moving_mean_initializer = moving_mean_initializer,
                              moving_variance_initializer = moving_variance_initializer,
                              beta_regularizer = beta_regularizer,
                              gamma_regularizer = gamma_regularizer,
                              beta_constraint = beta_constraint,
                              gamma_constraint = gamma_constraint,
                              renorm = renorm,
                              renorm_clipping = renorm_clipping,
                              renorm_momentum = renorm_momentum,
                              fused = fused,
                              trainable = trainable,
                              name = name,
                              dtype = inputs.dtype.base_dtype,
                              _reuse = reuse,
                              _scope = name)
    return layer.apply(inputs, training = training)

################ Deprecated #############
# This function computes the gradient of the binary weights
'''
def compute_grads(loss, opt):
    gradients_list  = []
    parameters_list = []


    for layer in all_layers:
        for var in layer.trainable_variables:
            if not var.name.endswith('kernel:0'):
                param = var
            else:
                param = layer.b_kernel

            #if len(params) > 0:
            #compute gradients for params
            grad = tf.gradients(loss, param)  # if len(param)=n, return [grad[0],...,grad[n-1]]

            #process gradients
            if var.name.endswith('gamma:0') or var.name.endswith('beta:0'): # for BN
                clipped_gradients = grad[0]
            else:
                clipped_gradients = clipping_scaling(param, grad[0], layer.W_LR_scale, layer.H)


            gradients_list.append(clipped_gradients)
            parameters_list.append(var)      # Note: update real value weight

    print(gradients_list)
    print(parameters_list)
    return zip(gradients_list, parameters_list)


# This functions clips the weights after the parameter update
def clipping_scaling(weight, grad, W_LR_scale, H):
    print("W_LR_scale = "+str(W_LR_scale))
    print("H = "+str(H))
    #W_LR_scale為learning_rate擴大倍數，updates[param]為更新後的權重，param為未更新權重
    #weight_scale = weight + W_LR_scale*(tf.subtract(grad, weight))
    weight_clip = tf.clip_by_value(grad, -H, H)

    return weight_clip
'''