# coding=UTF-8
import tensorflow as tf
from tensorflow.python.framework import tensor_shape, ops
from tensorflow.python.ops import standard_ops, nn, variable_scope, math_ops, control_flow_ops
from tensorflow.python.eager import context
from tensorflow.python.training import optimizer, training_ops
import numpy as np

# Warning: if you have a @property getter/setter function in a class, must inherit from object class

all_layers = []

def hard_sigmoid(x):
    return tf.clip_by_value((x + 1.)/2., 0, 1)

def round_through(x):
    '''Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    a op that behave as f(x) in forward mode,
    but as g(x) in the backward mode.
    '''
    rounded = tf.round(x)
    return x + tf.stop_gradient(rounded-x)

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
        #Wb = tf.round(Wb)

        # 0 or 1 -> -1 or 1
        #Wb = tf.where(tf.equal(Wb, 1.0), tf.ones_like(W), -tf.ones_like(W))  # cant differential
        Wb = H * binary_tanh_unit(W / H)

    return Wb


class Dense_BinaryLayer(tf.layers.Dense):
    def __init__(self, output_dim,
               activation = None,
               use_bias = True,
               binary = True, stochastic = True, H = 1., W_LR_scale="Glorot",
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
        print(num_units)

        if self.H == "Glorot":
            self.H = np.float32(np.sqrt(1.5 / (num_inputs + num_units)))   # weight init method
        self.W_LR_scale = np.float32(1. / np.sqrt(1.5 / (num_inputs + num_units))) # each layer learning rate
        print("H = ", self.H)
        print("LR scale = ", self.W_LR_scale)

        self.kernel_initializer = tf.random_uniform_initializer(-self.H, self.H)
        self.kernel_constraint = lambda w: tf.clip_by_value(w, -self.H, self.H)

        '''
        self.b_kernel = self.add_variable('binary_weight',
                                    shape=[input_shape[-1], self.units],
                                    initializer=self.kernel_initializer,
                                    regularizer=None,
                                    constraint=None,
                                    dtype=self.dtype,
                                    trainable=False)  # add_variable must execute before call build()
        '''
        self.b_kernel = self.add_variable('binary_weight',
                                    shape=[input_shape[-1], self.units],
                                    initializer=tf.random_uniform_initializer(-self.H, self.H),
                                    regularizer=None,
                                    constraint=None,
                                    dtype=self.dtype,
                                    trainable=False)

        super(Dense_BinaryLayer, self).build(input_shape)

        #tf.add_to_collection('real', self.trainable_variables)
        tf.add_to_collection(self.name + '_binary', self.kernel)  # layer-wise group
        tf.add_to_collection('binary', self.kernel)               # global group


    def call(self, inputs):
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        shape = inputs.get_shape().as_list()

        # binarization weight
        self.b_kernel = binarization(self.kernel, self.H)
        #r_kernel = self.kernel
        #self.kernel = self.b_kernel

        print("shape: ", len(shape))
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
    binary = True, stochastic = True, H=1., W_LR_scale="Glorot",
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
                binary = binary, stochastic = stochastic, H = H, W_LR_scale = W_LR_scale,
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
               binary = True, stochastic = True, H = 1., W_LR_scale = "Glorot",
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

        self.binary = binary
        self.stochastic = stochastic

        self.H = H
        self.W_LR_scale = W_LR_scale

        all_layers.append(self)

    def build(self, input_shape):
        num_inputs = np.prod(self.kernel_size) * tensor_shape.TensorShape(input_shape).as_list()[3]
        num_units = np.prod(self.kernel_size) * self.filters

        if self.H == "Glorot":
            self.H = np.float32(np.sqrt(1.5 / (num_inputs + num_units)))   # weight init method
        self.W_LR_scale = np.float32(1. / np.sqrt(1.5 / (num_inputs + num_units))) # each layer learning rate
        print("H = ", self.H)
        print("LR scale = ", self.W_LR_scale)

        self.kernel_initializer = tf.random_uniform_initializer(-self.H, self.H)
        self.kernel_constraint = lambda w: tf.clip_by_value(w, -self.H, self.H)

        self.b_kernel = 0  # add_variable must execute before call build()

        super(Conv2D_BinaryLayer, self).build(input_shape)

        tf.add_to_collection(self.name + '_binary', self.kernel)  # layer-wise group
        tf.add_to_collection('binary', self.kernel)

    def call(self, inputs):
        # binarization weight
        self.b_kernel = binarization(self.kernel, self.H)

        outputs = self._convolution_op(inputs, self.b_kernel)

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
           kernel_num,
           kernel_size,
           strides = (1, 1),
           padding = 'valid',
           data_format = 'channels_last',
           dilation_rate = (1, 1),
           activation = None,
           use_bias = True,
           binary = True, stochastic = True, H=1., W_LR_scale="Glorot",
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
                kernel_num = kernel_num,
                kernel_size = kernel_size,
                strides = strides,
                padding = padding,
                data_format = data_format,
                dilation_rate = dilation_rate,
                activation = activation,
                use_bias = use_bias,
                binary = binary, stochastic = stochastic, H = H, W_LR_scale = W_LR_scale,
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
      #all_layers.append(self)

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

class AdamOptimizer(optimizer.Optimizer):
    """Optimizer that implements the Adam algorithm.
    See [Kingma et al., 2014](http://arxiv.org/abs/1412.6980)
    ([pdf](http://arxiv.org/pdf/1412.6980.pdf)).
    """

    def __init__(self, weight_scale, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 use_locking=False, name="Adam"):
        super(AdamOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        # BNN weight scale factor
        self._weight_scale = weight_scale

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None

        # Variables to accumulate the powers of the beta parameters.
        # Created in _create_slots when we know the variables to optimize.
        self._beta1_power = None
        self._beta2_power = None

        # Created in SparseApply if needed.
        self._updated_lr = None

    def _get_beta_accumulators(self):
        return self._beta1_power, self._beta2_power

    def _non_slot_variables(self):
        return self._get_beta_accumulators()

    def _create_slots(self, var_list):
        first_var = min(var_list, key=lambda x: x.name)

        create_new = self._beta1_power is None
        if not create_new and context.in_graph_mode():
            create_new = (self._beta1_power.graph is not first_var.graph)

        if create_new:
            with ops.colocate_with(first_var):
                self._beta1_power = variable_scope.variable(self._beta1,
                                                            name="beta1_power",
                                                            trainable=False)
                self._beta2_power = variable_scope.variable(self._beta2,
                                                            name="beta2_power",
                                                            trainable=False)
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")

    def _apply_dense(self, grad, var):
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        # for BNN kernel
        # origin version clipping weight method is new_w = old_w + scale*(new_w - old_w)
        # and adam update function is new_w = old_w - lr_t * m_t / (sqrt(v_t) + epsilon)
        # so subtitute adam function into weight clipping
        # new_w = old_w - (scale * lr_t * m_t) / (sqrt(v_t) + epsilon)
        scale = self._weight_scale[ var.name ] / 4

        return training_ops.apply_adam(
            var, m, v,
            math_ops.cast(self._beta1_power, var.dtype.base_dtype),
            math_ops.cast(self._beta2_power, var.dtype.base_dtype),
            math_ops.cast(self._lr_t * scale, var.dtype.base_dtype),
            math_ops.cast(self._beta1_t, var.dtype.base_dtype),
            math_ops.cast(self._beta2_t, var.dtype.base_dtype),
            math_ops.cast(self._epsilon_t, var.dtype.base_dtype),
            grad, use_locking=self._use_locking).op

    def _resource_apply_dense(self, grad, var):
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        return training_ops.resource_apply_adam(
            var.handle, m.handle, v.handle,
            math_ops.cast(self._beta1_power, grad.dtype.base_dtype),
            math_ops.cast(self._beta2_power, grad.dtype.base_dtype),
            math_ops.cast(self._lr_t, grad.dtype.base_dtype),
            math_ops.cast(self._beta1_t, grad.dtype.base_dtype),
            math_ops.cast(self._beta2_t, grad.dtype.base_dtype),
            math_ops.cast(self._epsilon_t, grad.dtype.base_dtype),
            grad, use_locking=self._use_locking)

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(self._beta2_power, var.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
        lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))
        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(m, m * beta1_t,
                               use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = scatter_add(m, indices, m_scaled_g_values)
        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad * grad) * (1 - beta2_t)
        v_t = state_ops.assign(v, v * beta2_t, use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = scatter_add(v, indices, v_scaled_g_values)
        v_sqrt = math_ops.sqrt(v_t)
        var_update = state_ops.assign_sub(var,
                                          lr * m_t / (v_sqrt + epsilon_t),
                                          use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
          grad.values, var, grad.indices,
          lambda x, i, v: state_ops.scatter_add(  # pylint: disable=g-long-lambda
              x, i, v, use_locking=self._use_locking))

    def _resource_scatter_add(self, x, i, v):
        with ops.control_dependencies(
            [resource_variable_ops.resource_scatter_add(
                x.handle, i, v)]):
            return x.value()

    def _resource_apply_sparse(self, grad, var, indices):
        return self._apply_sparse_shared(
            grad, var, indices, self._resource_scatter_add)

    def _finish(self, update_ops, name_scope):
        # Update the power accumulators.
        with ops.control_dependencies(update_ops):
            with ops.colocate_with(self._beta1_power):
                update_beta1 = self._beta1_power.assign(
                    self._beta1_power * self._beta1_t,
                    use_locking=self._use_locking)
                update_beta2 = self._beta2_power.assign(
                    self._beta2_power * self._beta2_t,
                    use_locking=self._use_locking)
        return control_flow_ops.group(*update_ops + [update_beta1, update_beta2],
                                      name=name_scope)

def get_all_layers():
    return all_layers;

def get_all_LR_scale():
    return {layer.kernel.name: layer.W_LR_scale for layer in get_all_layers()}

# This function computes the gradient of the binary weights
def compute_grads(loss, opt):
    layers = get_all_layers()
    grads_list = []
    update_weights = []

    for layer in layers:

        # refer to self.params[self.W]=set(['binary'])
        # The list can optionally be filtered by specifying tags as keyword arguments.
        # For example,
        #``trainable=True`` will only return trainable parameters, and
        #``regularizable=True`` will only return parameters that can be regularized
        # function return, e.g. [W, b] for dense layer
        params = tf.get_collection(layer.name + "_binary")
        if params:
            # print(params[0].name)
            # theano.grad(cost, wrt) -> d(cost)/d(wrt)
            # wrt – with respect to which we want gradients
            # http://blog.csdn.net/shouhuxianjian/article/details/46517143
            # http://blog.csdn.net/qq_33232071/article/details/52806630
            #grad = opt.compute_gradients(loss, layer.b_kernel)  # origin version
            grad = opt.compute_gradients(loss, params[0])        # modify
            print("grad: ", grad)
            grads_list.append( grad[0][0] )
            update_weights.extend( params )

    print(grads_list)
    print(update_weights)
    return zip(grads_list, update_weights)
