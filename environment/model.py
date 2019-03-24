import keras.layers as l
from keras.models import Sequential
from keras.optimizers import Adam

from keras import backend as K
from keras.layers import Layer
from keras.regularizers import Regularizer
import numpy as np

from keras.initializers import RandomUniform, Constant, TruncatedNormal, Zeros
from keras.constraints import NonNeg, MinMaxNorm

import tensorflow as tf

from math import pi

activation = 'elu'
momentum = 0.9


class DiscretizationLayerWide(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(DiscretizationLayerWide, self).__init__(**kwargs)

    def build(self, input_shape):
        u = -6
        l = 6
        initer = [np.linspace(l, u, self.output_dim).reshape(1, -1) for _ in range(input_shape[1])]
        initer = np.concatenate(initer, axis=0)
        init = Constant(initer)

        bias_initializer = Constant(0)
        width_val = 3. * float(u - l) / input_shape[1]
        super(DiscretizationLayerWide, self).build(input_shape)
        self.bins = self.add_weight(name='bins',
                                    shape=(input_shape[1], self.output_dim),
                                    initializer=init,
                                    trainable=True)

        self.widths = self.add_weight(name='widths',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=TruncatedNormal(width_val, width_val / 4),
                                      constraint=NonNeg(),
                                      trainable=True)

        self.biases = self.add_weight(name='biases',
                                      shape=(input_shape[1], self.output_dim,),
                                      initializer=bias_initializer,
                                      trainable=True)

        self.dense_weight = self.add_weight(name='w',
                                            shape=(input_shape[1], self.output_dim),
                                            initializer='glorot_uniform',
                                            trainable=True)

        self.dense_bias = self.add_weight(name='b',
                                          shape=(input_shape[1],),
                                          initializer=Zeros(),
                                          trainable=True)

        self.built = True

    def call(self, inputs, **kwargs):
        input = tf.expand_dims(inputs, -1)
        bins = self.biases - tf.abs(input - self.bins) * self.widths
        bins = tf.nn.leaky_relu(bins)
        bins2prob = tf.nn.softmax(bins)
        x = bins2prob * self.dense_weight
        x = tf.reduce_sum(x, axis=2) + self.dense_bias
        x = tf.nn.tanh(x)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape


def create_common_model(config, input_dim, multiplier=2):
    if not config['use common']:
        return None
    multiplier_common = config['multiplier common']
    layers_common = config['layers common']

    dropout_rate = config['dropout rate']

    common_model = Sequential()
    common_model.add(l.Dense(input_dim * multiplier_common,
                             input_shape=(multiplier * input_dim,), activation=activation))
    if config['use batch normalization']:
        common_model.add(l.BatchNormalization(momentum=momentum))
    for _ in range(layers_common):
        common_model.add(l.Dropout(dropout_rate))
        common_model.add(l.Dense(input_dim * multiplier_common, activation=activation))
        #if config['use batch normalization']:
        #    common_model.add(l.BatchNormalization(momentum=momentum))

    return common_model


def create_eviction_model(config, input_dim, common_model, multiplier=2):
    wing_size = config["wing size"]
    last_dim = wing_size * 2 + 1

    dropout_rate = config['dropout rate']

    multiplier_each = config['multiplier each']
    layers_each = config['layers each']

    model_eviction = Sequential()
    if config['use common']:
        model_eviction.add(common_model)
    else:
        model_eviction.add(l.Dense(input_dim * multiplier_each,
                                   input_shape=(multiplier * input_dim,), activation=activation))
        if config['use batch normalization']:
            model_eviction.add(l.BatchNormalization(momentum=momentum))

    for i in range(layers_each):
        model_eviction.add(l.Dropout(dropout_rate))
        model_eviction.add(l.Dense(input_dim * int(multiplier_each * (layers_each - i) / layers_each),
                                   activation=activation))
        #if config['use batch normalization']:
        #    model_eviction.add(l.BatchNormalization(momentum=momentum))

    model_eviction.add(l.Dropout(dropout_rate))
    activation_last = 'softmax'
    if config['mc']:
        activation_last = 'sigmoid'
    model_eviction.add(l.Dense(last_dim, activation=activation_last))

    optimizer = Adam(lr=config['eviction lr'])

    loss = 'categorical_crossentropy'
    metrics = ['accuracy']
    if config['mc']:
        loss = 'mse'
        metrics = ['mse']
    model_eviction.compile(optimizer, loss=loss, metrics=metrics)

    return model_eviction


def compile_model(model, config, mtype):
    lr = None
    if mtype == 'E':
        lr = config['eviction lr']
    if mtype == 'A':
        lr = config['admission lr']
    assert lr is not None
    optimizer = Adam(lr=lr)
    loss = 'mse'
    metrics = ['mse']
    if config['mc']:
        loss = 'mse'
        metrics = ['mse']
    model.compile(optimizer, loss=loss, metrics=metrics)
    return model


def create_admission_model(config, input_dim, common_model, multiplier=2):

    dropout_rate = config['dropout rate']

    multiplier_each = config['multiplier each']
    layers_each = config['layers each']

    use_discretization = config['use discretization']

    model_admission = Sequential()
    if not config['use common']:
        if use_discretization:
            model_admission.add(DiscretizationLayerWide(200, input_shape=(multiplier * input_dim,)))
            model_admission.add(l.BatchNormalization())
            input_dim *= 10
        else:
            model_admission.add(
                l.Dense(input_dim * multiplier_each, input_shape=(multiplier * input_dim,), activation=activation))
        if config['use batch normalization']:
            model_admission.add(l.BatchNormalization())
    else:
        model_admission.add(common_model)

    for i in range(layers_each):
        model_admission.add(l.Dropout(dropout_rate))
        model_admission.add(l.Dense(input_dim * int(multiplier_each * (layers_each - i) / layers_each),
                                    activation=activation))
        #if config['use batch normalization']:
        #    model_admission.add(l.BatchNormalization(momentum=momentum))
    model_admission.add(l.Dropout(dropout_rate))
    activation_last = 'softmax'
    if config['mc']:
        activation_last = 'sigmoid'
    model_admission.add(l.Dense(2, activation=activation_last))

    optimizer = Adam(lr=config['admission lr'])

    loss = 'categorical_crossentropy'
    metrics = ['accuracy']
    if config['mc']:
        loss = 'mse'
        metrics = ['mse']
    model_admission.compile(optimizer, loss=loss, metrics=metrics)

    return model_admission


def create_models(config, input_dim, multiplier=2):
    wing_size = config["wing size"]
    last_dim = wing_size * 2 + 1

    if config['use common']:
        common_model = create_common_model(config, input_dim, multiplier)
    else:
        common_model = None
    model_admission = create_admission_model(config, input_dim, common_model, multiplier)
    model_eviction = create_eviction_model(config, input_dim, common_model, multiplier)
    return model_admission, model_eviction, common_model, last_dim
