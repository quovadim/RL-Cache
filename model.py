import keras.layers as l
from keras.models import Sequential
from keras.optimizers import Adam


def create_common_model(config, input_dim):
    if not config['use common']:
        return None
    multiplier_common = config['multiplier common']
    layers_common = config['layers common']

    dropout_rate = config['dropout rate']

    common_model = Sequential()
    common_model.add(l.Dense(input_dim * multiplier_common, input_shape=(2 * input_dim,), activation='elu'))
    if config['use batch normalization']:
        common_model.add(l.BatchNormalization())
    for _ in range(layers_common):
        common_model.add(l.Dropout(dropout_rate))
        common_model.add(l.Dense(input_dim * multiplier_common, activation='elu'))
        if config['use batch normalization']:
            common_model.add(l.BatchNormalization())

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
        model_eviction.add(l.Dense(input_dim * multiplier_each, input_shape=(2 * input_dim,), activation='elu'))
        if config['use batch normalization']:
            model_eviction.add(l.BatchNormalization())

    for i in range(layers_each):
        model_eviction.add(l.Dropout(dropout_rate))
        model_eviction.add(l.Dense(input_dim * int(multiplier_each * (layers_each - i) / layers_each),
                                   activation='elu'))
        if config['use batch normalization']:
            model_eviction.add(l.BatchNormalization())

    model_eviction.add(l.Dropout(dropout_rate))
    model_eviction.add(l.Dense(last_dim, activation='softmax'))

    optimizer = Adam(lr=config['eviction lr'])

    model_eviction.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model_eviction


def create_admission_model(config, input_dim, common_model):

    dropout_rate = config['dropout rate']

    multiplier_each = config['multiplier each']
    layers_each = config['layers each']

    model_admission = Sequential()
    if not config['use common']:
        model_admission.add(l.Dense(input_dim * multiplier_each, input_shape=(2 * input_dim,), activation='elu'))
        if config['use batch normalization']:
            model_admission.add(l.BatchNormalization())
    else:
        model_admission.add(common_model)

    for i in range(layers_each):
        model_admission.add(l.Dropout(dropout_rate))
        model_admission.add(l.Dense(input_dim * int(multiplier_each * (layers_each - i) / layers_each),
                                    activation='elu'))
        if config['use batch normalization']:
            model_admission.add(l.BatchNormalization())
    model_admission.add(l.Dropout(dropout_rate))
    model_admission.add(l.Dense(2, activation='softmax'))

    optimizer = Adam(lr=config['admission lr'])

    model_admission.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model_admission
