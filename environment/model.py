import keras.layers as l
from keras.models import Sequential
from keras.optimizers import Adam

from keras import backend as K
from keras.layers import Layer
from keras.regularizers import Regularizer
import numpy as np

from keras.initializers import RandomUniform, Constant
from keras.constraints import NonNeg, MinMaxNorm

from math import pi

activation = 'elu'
momentum = 0.9


class DiscretizationLayer(Layer):
    def __init__(self, output_dim, init_means, init_sigmas, distribution='Normal', normalization='none', **kwargs):
        self.output_dim = output_dim
        self.init_means = init_means
        self.init_sigmas = init_sigmas
        self.distribution = distribution
        self.normalization = normalization
        super(DiscretizationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(DiscretizationLayer, self).build(input_shape)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=Constant(self.init_means),
                                      trainable=True)

        self.sigmas = self.add_weight(name='sigma',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=Constant(self.init_sigmas),
                                      constraint=NonNeg(),
                                      trainable=True)

        if self.normalization == 'softmax':
            self.temperature = self.add_weight(name='temperature',
                                      shape=(input_shape[1], 1),
                                      initializer=RandomUniform(0.5 - 1./input_shape[1], 0.5 + 1./input_shape[1]),
                                      constraint=MinMaxNorm(min_value=1e-2, max_value=1, axis=0),
                                      trainable=True)

        self.built = True

    def call(self, inputs, **kwargs):
        source_matrix = K.repeat(inputs, self.output_dim)

        source_matrix = K.permute_dimensions(source_matrix, (0, 2, 1))

        if self.distribution == 'Laplace':

            laplace_distr = -1. * K.abs(source_matrix - self.kernel)

            sigma_abs = K.abs(self.sigmas)
            laplace_distr = K.exp(laplace_distr / (1e-5 + sigma_abs))

            source_matrix = laplace_distr / (1e-5 + 2 * sigma_abs)

        if self.distribution == 'Normal':

            normal_distr = -1. * K.square(source_matrix - self.kernel)

            sigma_square = self.sigmas * self.sigmas
            normal_distr = K.exp(normal_distr / (1e-5 + 2 * sigma_square))

            source_matrix = normal_distr / (1e-5 + K.sqrt(sigma_square * 2 * pi))

        if self.distribution == 'Grumbel':
            self.sigmas += 0.5
            z = (source_matrix - self.kernel) / (1e-5 + self.sigmas)
            exp_z = K.exp(z)
            z = -1. * (z + exp_z)
            z = K.exp(z) / (1e-5 + self.sigmas)
            source_matrix = z

        normed = source_matrix

        if self.normalization == 'softmax':
            normed = normed / self.temperature
            normed = K.softmax(normed, axis=2)
        if self.normalization == 'l1':
            normed += 1e-5
            normed = normed / (1e-5 + K.sum(K.abs(normed), axis=2, keepdims=True))

        return normed

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim,)

    def get_config(self):
        config = {'init_means': self.init_means,
                  'init_sigmas': self.init_sigmas,
                  'normalization': self.normalization,
                  'distribution': self.distribution}
        base_config = super(DiscretizationLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class EntropyRegularizer(Regularizer):
    def __init__(self, p=0.):
        self.p = K.cast_to_floatx(p)

    def __call__(self, x):
        log_data = K.log(1e-10 + K.abs(x))
        entropy = K.abs(K.sum(log_data * K.abs(x), axis=2))
        return self.p * K.mean(entropy)

    def get_config(self):
        return {'p': float(self.p)}


class EntropyRegularization(Layer):
    def __init__(self, p=0., **kwargs):
        super(EntropyRegularization, self).__init__(**kwargs)
        self.supports_masking = True
        self.p = p
        self.activity_regularizer = EntropyRegularizer(self.p)

    def get_config(self):
        config = {'p': self.p}
        base_config = super(EntropyRegularization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

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
            lsz = 10
            initial_values = [np.random.uniform(-3, 3, lsz) for a, b in config['init vals']]
            initial_values += initial_values
            initial_values = np.array(initial_values)
            initial_sigmas = [np.random.uniform(0.1, 0.3, lsz) for a, b in config['init vals']]
            initial_sigmas += initial_sigmas
            initial_sigmas = np.array(initial_sigmas)
            model_admission.add(DiscretizationLayer(lsz, initial_values, initial_sigmas,
                                                    input_shape=(multiplier * input_dim,)))
            model_admission.add(l.Flatten())
            input_dim *= lsz
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
