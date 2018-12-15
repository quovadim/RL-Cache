import keras.layers as l
from keras.models import Sequential
from keras.optimizers import Adam

activation = 'elu'
momentum = 0.9


def create_common_model(config, input_dim):
    if not config['use common']:
        return None
    multiplier_common = config['multiplier common']
    layers_common = config['layers common']

    dropout_rate = config['dropout rate']

    common_model = Sequential()
    common_model.add(l.Dense(input_dim * multiplier_common, input_shape=(2 * input_dim,), activation=activation))
    if config['use batch normalization']:
        common_model.add(l.BatchNormalization(momentum=momentum))
    for _ in range(layers_common):
        common_model.add(l.Dropout(dropout_rate))
        common_model.add(l.Dense(input_dim * multiplier_common, activation=activation))
        #if config['use batch normalization']:
        #    common_model.add(l.BatchNormalization(momentum=momentum))

    return common_model


def create_eviction_model(config, input_dim, common_model):
    wing_size = config["wing size"]
    last_dim = wing_size * 2 + 1

    dropout_rate = config['dropout rate']

    multiplier_each = config['multiplier each']
    layers_each = config['layers each']

    model_eviction = Sequential()
    if config['use common']:
        model_eviction.add(common_model)
    else:
        model_eviction.add(l.Dense(input_dim * multiplier_each, input_shape=(2 * input_dim,), activation=activation))
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
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']
    if config['mc']:
        loss = 'mse'
        metrics = ['mse']
    model.compile(optimizer, loss=loss, metrics=metrics)
    return model


def create_admission_model(config, input_dim, common_model):

    dropout_rate = config['dropout rate']

    multiplier_each = config['multiplier each']
    layers_each = config['layers each']

    model_admission = Sequential()
    if not config['use common']:
        model_admission.add(l.Dense(input_dim * multiplier_each, input_shape=(2 * input_dim,), activation=activation))
        if config['use batch normalization']:
            model_admission.add(l.BatchNormalization(momentum=momentum))
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


def create_models(config, input_dim):
    wing_size = config["wing size"]
    last_dim = wing_size * 2 + 1

    if config['use common']:
        common_model = create_common_model(config, input_dim)
    else:
        common_model = None
    model_admission = create_admission_model(config, input_dim, common_model)
    model_eviction = create_eviction_model(config, input_dim, common_model)
    return model_admission, model_eviction, common_model, last_dim
