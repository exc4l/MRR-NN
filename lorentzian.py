import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers, initializers
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras import constraints
import numpy as np
from scipy.optimize import root_scalar, fsolve


class MinMaxClipping(tf.keras.constraints.Constraint):
    def __init__(self, minval, maxval):
        self.min_val = minval
        self.max_val = maxval

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_val, self.max_val)


class Lorentzian(keras.layers.Layer):
    def __init__(
        self,
        units,
        gamma=5,
        use_bias=False,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        train_scale_output=None,
        output_splitter=None,
        kernel_constraint=None,
        bias_constraint=None,
        sca_out_constraint=MinMaxClipping(0.1, 1),
        **kwargs,
    ):
        super(Lorentzian, self).__init__(
            activity_regularizer=activity_regularizer, **kwargs
        )
        self.units = int(units) if not isinstance(units, int) else units
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.sca_out_initializer = keras.initializers.get("ones")
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.sca_out_constraint = constraints.get(sca_out_constraint)
        self.use_bias = use_bias
        self.train_scale_output = train_scale_output
        self.output_splitter = output_splitter

        self.gamma = tf.Variable(gamma, trainable=False, dtype="float32")
        self.intensity = tf.Variable(1, trainable=False, dtype="float32")
        self.offset = tf.Variable(0, trainable=False, dtype="float32")

        self.input_spec = InputSpec(min_ndim=2)

    def build(self, input_shape):
        dtype = "float32"
        #         if not (dtype.is_floating or dtype.is_complex):
        #           raise TypeError('Unable to build `Dense` layer with non-floating point '
        #                           'dtype %s' % (dtype,))

        input_shape = tensor_shape.TensorShape(input_shape)
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError(
                "The last dimension of the inputs to `Lorentzian` "
                "should be defined. Found `None`."
            )
        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
        self.kernel = self.add_weight(
            "kernel",
            shape=[last_dim, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                "bias",
                shape=[
                    self.units,
                ],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True,
            )
        if self.train_scale_output:
            self.sca_out = self.add_weight(
                "sca_out",
                shape=[
                    self.units,
                ],
                initializer=self.sca_out_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.sca_out_constraint,
                dtype=self.dtype,
                trainable=True,
            )
        self.built = True

    def call(self, inputs):
        # a = tf.matmul(inputs, self.kernel) + self.bias
        # gsq = tf.square(self.gamma)
        # top = self.intensity * gsq
        # # bottom = (tf.matmul(inputs, self.kernel) + self.bias) + gsq
        # bottom = a + gsq
        # return tf.divide(top, bottom)
        # return tf.matmul(inputs, self.kernel) + self.bias
        gsq = tf.square(self.gamma)
        top = self.intensity * gsq
        a = tf.square(self.kernel)
        bottom = a + gsq
        intens = self.intensity
        if self.output_splitter:
            intens = self.intensity / self.output_splitter
        if self.train_scale_output:
            intens = tf.divide(intens, self.sca_out)
            return intens * tf.matmul(inputs, tf.divide(top, bottom))
        return intens * tf.matmul(inputs, tf.divide(top, bottom))


class MMRelu(keras.layers.Layer):
    def __init__(
        self,
        units,
        use_bias=False,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        train_scale_output=True,
        output_splitter=None,
        kernel_constraint=MinMaxClipping(0.01, 1),
        bias_constraint=None,
        sca_out_constraint=MinMaxClipping(0.01, 1),
        **kwargs,
    ):
        super(MMRelu, self).__init__(
            activity_regularizer=activity_regularizer, **kwargs
        )
        self.units = int(units) if not isinstance(units, int) else units
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.sca_out_initializer = keras.initializers.get("ones")
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.sca_out_constraint = constraints.get(sca_out_constraint)
        self.use_bias = use_bias
        self.train_scale_output = train_scale_output
        self.output_splitter = output_splitter

        self.input_spec = InputSpec(min_ndim=2)

    def build(self, input_shape):
        dtype = "float32"
        #         if not (dtype.is_floating or dtype.is_complex):
        #           raise TypeError('Unable to build `Dense` layer with non-floating point '
        #                           'dtype %s' % (dtype,))

        input_shape = tensor_shape.TensorShape(input_shape)
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError(
                "The last dimension of the inputs to `Lorentzian` "
                "should be defined. Found `None`."
            )
        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
        self.kernel = self.add_weight(
            "kernel",
            shape=[last_dim, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                "bias",
                shape=[
                    self.units,
                ],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True,
            )
        if self.train_scale_output:
            self.sca_out = self.add_weight(
                "sca_out",
                shape=[
                    self.units,
                ],
                initializer=self.sca_out_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.sca_out_constraint,
                dtype=self.dtype,
                trainable=True,
            )
        self.built = True

    def call(self, inputs):
        intens = tf.constant([1], dtype="float32")
        if self.output_splitter:
            intens = intens / self.output_splitter
        if self.train_scale_output:
            intens = tf.divide(intens, self.sca_out)
            return intens * tf.keras.activations.relu(tf.matmul(inputs, self.kernel))
        return intens * tf.keras.activations.relu(tf.matmul(inputs, self.kernel))


def lorz(x, gamma=5):
    I = 1
    gamma = gamma
    x0 = 0
    gsq = np.square(gamma)
    top = np.multiply(I, gsq)
    lbt = np.square(x - x0)
    bottom = lbt + gsq
    y = np.divide(top, bottom)
    return y


def get_lorzkernel(kernel, gamma):
    lorzkernel = np.zeros(kernel.shape, dtype="float32")
    for x in np.ndindex(kernel.shape):
        ele = kernel[x]
        if ele == 0:
            lorzkernel[x] = 100000
        else:
            sol = root_scalar(
                lambda x: lorz(x, gamma) - ele, method="brenth", bracket=[0, 150]
            )
            lorzkernel[x] = sol.root
    return lorzkernel


def create_lorentz_from_mrrelu(model, gamma=5):
    nmodel = tf.keras.Sequential()
    for idx in range(len(model.layers)):
        layer = model.layers[idx]
        if "flatten" in layer.name:
            inp_sh = tuple(c for c in layer.input_shape if c)
            nmodel.add(tf.keras.layers.Flatten(input_shape=inp_sh))
        if "mm_relu" in layer.name:
            if len(layer.get_weights()) > 1:
                nmodel.add(
                    Lorentzian(
                        layer.get_weights()[0].shape[1],
                        gamma=gamma,
                        train_scale_output=True,
                    )
                )
                old_weights = nmodel.layers[idx].get_weights()
                old_weights[0] = get_lorzkernel(layer.get_weights()[0], gamma)
                old_weights[1] = layer.get_weights()[1]
            else:
                nmodel.add(
                    Lorentzian(
                        layer.get_weights()[0].shape[1],
                        gamma=gamma,
                        train_scale_output=None,
                    )
                )
                old_weights = nmodel.layers[idx].get_weights()
                old_weights[0] = get_lorzkernel(layer.get_weights()[0], gamma)
            nmodel.layers[idx].set_weights(old_weights)
    return nmodel


def norm_to_wavelength(wvl, offset):
    if wvl > 10:
        return offset * 1e9
    if wvl < 10:
        if wvl > 1:
            return offset * 1e6
        if wvl < 1:
            return offset * 1e3


def offset_to_wavelength(wvl, noff):
    if noff > 100:
        return 0
    else:
        return wvl - noff


def get_ring_specs(model, wvl, gamma, sensitivity=1e-9 / 5e-3, filename="layer.txt"):
    """Only Lorentzian Layers work"""
    # sensitivity meint wie sehr sich der ring verschiebt pro mW Heizleistung. In meiner BA waren das etwa 1nm verschiebung pro 5mW.
    with open(filename, "w", encoding="utf-8") as wr:
        for idx in range(len(model.layers)):
            layer = model.layers[idx]
            if "flatten" in layer.name:
                wr.write("\n\n Flatten \n\n")
                continue
            if "lorentzian" in layer.name:
                wr.write(f"\n\n Layer Lorentzian {idx}: \n\n")
                gam_old = model.layers[idx].get_weights()[2]
                gam_new = gamma
                offset = model.layers[idx].get_weights()[0] * gam_new / gam_old
                #                 np.savetxt("weight.csv", offset, fmt="%s", delimiter=",")
                #                 with open("layer.txt", "w", encoding="utf-8") as wr:
                for nidx in np.ndindex(offset.shape):
                    noff = norm_to_wavelength(wvl, offset[nidx])
                    heat = noff / norm_to_wavelength(wvl, sensitivity) * 1e3
                    wr.write(
                        f"{nidx}, {wvl}nm, {offset_to_wavelength(wvl,noff):.6f}nm, {gamma}, {noff:.6f}nm, {heat:.6f}mW\n"
                    )
