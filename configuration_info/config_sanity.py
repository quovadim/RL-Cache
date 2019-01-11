from environment.environment_aux import load_json, name2class, class2name
from feature.extractor import PacketFeaturer
from environment.model import *
import os.path
from copy import deepcopy
from filestructure import *


error_levels = {
    0: '\033[1m\033[91mCRITICAL\033[0m',
    1: '\033[1m\033[93mWARNING\033[0m',
    2: '\033[1m\033[94mFIXED\033[0m',
    3: '\033[1m\033[92mOK\033[0m'
}


def check_range(name, value, interval_left, interval_right, tabulation, level, verbose, recommendation):
    condition_left = interval_left is not None and value < interval_left
    condition_right = interval_right is not None and value > interval_right
    value_type = 'obligatory' if not recommendation else 'recommended'
    if condition_left:
        print tabulation, error_levels[level], name, '=', value, 'is less than', value_type, interval_left
        return False

    if condition_right:
        print tabulation, error_levels[level], name, '=', value, 'is higher than', value_type, interval_right
        return False

    if verbose:
        if interval_left is None:
            interval_left = '-inf'
        if interval_right is None:
            interval_right = '+inf'
        print tabulation, error_levels[3], name, '=', value, \
            'is in', value_type, 'interval from', interval_left, 'to', interval_right

    return True


def check_existance(name, value, exists, tabulation, level, verbose, dead, directory=False):
    if not directory:
        file_exists = os.path.isfile(value)
    else:
        file_exists = os.path.isdir(value)
    real_status = 'exists' if file_exists else 'not exists'
    theoretical_status = 'exists' if exists else 'not exists'
    if exists != file_exists:
        print_needed = dead
        result = False
        lvl = level
    else:
        print_needed = verbose
        result = True
        lvl = 3

    if print_needed:
        print tabulation, error_levels[lvl], name, 'file', value, 'should', \
            theoretical_status, 'while it does', real_status

    return result


def apply_config(configuration_rules, configuration, tabulation):
    for key in configuration.keys():
        if key not in configuration_rules.keys():
            print tabulation, error_levels[0], 'unrecognized key', key
            return None
        rules = configuration_rules[key]
        if rules["interval"] is not None:
            l, r = rules["interval"]
            if not check_range(key, configuration[key], l, r, tabulation, 0, False, True):
                return None
        if rules["recommendation"] is not None:
            l, r = rules["recommendation"]
            check_range(key, configuration[key], l, r, tabulation, 1, False, True)
    keys_to_fill = [key for key in configuration_rules.keys() if key not in configuration.keys()]
    for key in keys_to_fill:
        rules = configuration_rules[key]
        if rules['required']:
            print tabulation, error_levels[0], 'key', key, 'missing'
            return None
        print tabulation, error_levels[2], 'to', key, 'assigned', rules['default']
        configuration[key] = rules['default']
    return configuration


def compare_statistics_dicts(d1, d2):
    c1 = deepcopy(d1)
    c2 = deepcopy(d2)
    if 'load' not in c1.keys() or 'save' not in c1.keys() or 'show stat' not in c1.keys():
        return False
    if 'load' not in c2.keys() or 'save' not in c2.keys() or 'show stat' not in c2.keys():
        return False
    c1['load'] = c2['load']
    c1['save'] = c2['save']
    c1['show stat'] = c2['show stat']
    return c1 == c2


def load_caching_algorithms(algorithms, tabulation, verbose):
    known_configs = [{}]
    known_names = ['']
    featurers = [PacketFeaturer(None, False)]

    known_admission_models = [None]
    known_eviction_models = [None]

    known_admission_names = ['']
    known_eviction_names = ['']

    common_models_mapping = {}

    models = {}

    for key in algorithms.keys():
        experiment = algorithms[key]
        class_info = name2class(key)
        if experiment is not None:
            target_index = -1
            model_config = check_model_config(experiment, tabulation=tabulation, verbose=verbose)
            #Check model and it's Statistics, based on this create featurer
            if model_config is None:
                return None
            if experiment in known_names:
                target_index = known_names.index(experiment)
            else:
                statistics_config = check_statistics_config(experiment, verbose=False)
                if statistics_config is None:
                    return None
                for i in range(len(known_configs)):
                    if compare_statistics_dicts(statistics_config, known_configs[i]):
                        target_index = i
                        break
                if target_index < 0:
                    target_index = len(known_names)
                    known_names.append(experiment)
                    known_configs.append(statistics_config)
                    featurers.append(PacketFeaturer(statistics_config, verbose=False))

            adm_model = 0
            evc_model = 0
            cm_model = None
            input_dim = featurers[target_index].dim
            if model_config['use common']:
                cm_model = create_common_model(model_config, input_dim)
            common_models_mapping[key] = cm_model
            if class_info['admission mode']:
                if experiment in known_admission_names:
                    adm_model = known_admission_names.index(experiment)
                else:
                    adm_model = len(known_admission_models)
                    model = create_admission_model(model_config, input_dim, cm_model)
                    try:
                        model.load_weights(get_admission_name(experiment))
                    except IOError:
                        pass
                    known_admission_models.append(model)
                    known_admission_names.append(experiment)
            if class_info['eviction mode']:
                if experiment in known_eviction_names:
                    evc_model = known_eviction_names.index(experiment)
                else:
                    evc_model = len(known_eviction_models)
                    model = create_eviction_model(model_config, input_dim, cm_model)
                    try:
                        model.load_weights(get_eviction_name(experiment))
                    except IOError:
                        pass
                    known_eviction_models.append(model)
                    known_eviction_names.append(experiment)
            models[key] = (adm_model, evc_model)
        else:
            models[key] = (0, 0)

    return featurers, known_admission_models, known_eviction_models, models, common_models_mapping


def check_statistics_config(experiment_name, tabulation='', verbose=True):

    filename = get_statistics_name(experiment_name)
    configuration_rules = load_json(get_configuration_rules('statistics.json'))

    if not check_existance('source', filename, True, tabulation, 0, verbose, True):
        return None

    tabulation = tabulation + ' ' + filename

    try:
        config = load_json(filename)
        if verbose:
            print tabulation, error_levels[3], 'File', filename, 'loaded'
    except:
        print tabulation, error_levels[0], "Error during loading", filename
        return None

    config = apply_config(configuration_rules, config, tabulation)

    if not config['usable names']:
        config['usable names'] = PacketFeaturer.log_features

    config['filename'] = get_intervals_name(experiment_name)

    if not config['save'] and not config['load']:
        print tabulation, error_levels[1], 'We would strongly recommend you to save or load the data'

    if not os.path.isfile(config['filename']) and config['load']:
        print tabulation, error_levels[1], 'File', config['filename'], 'does not exists, data will be generated'

    if os.path.isfile(config['filename']) and config['save'] and not config['load']:
        print tabulation, error_levels[1], 'File', config['filename'], 'will be overwritten'

    return config


def check_model_config(experiment_name, tabulation='', verbose=True):

    filename = get_model_name(experiment_name)
    configuration_rules = load_json(get_configuration_rules('model.json'))

    if not check_existance('source', filename, True, tabulation, 0, verbose, True):
        return None

    tabulation = tabulation + ' ' + filename

    try:
        config = load_json(filename)
        if verbose:
            print tabulation, error_levels[3], 'File', filename, 'loaded'
    except:
        print tabulation, error_levels[0], "Error during loading", filename
        return None

    config = apply_config(configuration_rules, config, tabulation)

    config['statistics'] = get_statistics_name(experiment_name)

    if not check_existance('statistics', config['statistics'], True, tabulation, 0, verbose, True):
        return None

    return config


def check_session_configuration(config, tabulation):

    configuration_rules = load_json(get_configuration_rules('session.json'))

    tabulation = tabulation + ' ' + 'session'

    config = apply_config(configuration_rules, config, tabulation)

    return config


def check_train_config(experiment_name, tabulation='', verbose=True):

    filename = get_train_name(experiment_name)

    if not check_existance('source', filename, True, tabulation, 0, verbose, True):
        return None

    tabulation = tabulation + ' ' + filename
    try:
        config = load_json(filename)
        if verbose:
            print tabulation, error_levels[3], 'File', filename, 'loaded'
    except:
        print tabulation, error_levels[0], "Error during loading", filename
        return None

    configuration_rules = load_json(get_configuration_rules('train.json'))
    config = apply_config(configuration_rules, config, tabulation)

    config["data folder"] = get_data_name(config["data"])
    config["train history"] = get_history_name(experiment_name)
    config["eviction path"] = get_eviction_name(experiment_name)
    config["admission path"] = get_admission_name(experiment_name)

    target_data = name2class(config['target'])
    target_data['size'] = config['cache size']
    config['special keys'] = class2name(target_data)

    algorithms_named = class2name(target_data)
    for key in config['algorithms']:
        key_data = name2class(key)
        key_data['size'] = config['cache size']
        algorithms_named += class2name(key_data)
    config['algorithms'] = algorithms_named

    if not check_existance('data folder', config['data folder'], True, tabulation, 0, verbose, True, directory=True):
        return None

    config['model'] = check_model_config(experiment_name, tabulation=tabulation, verbose=verbose)
    if config['model'] is None:
        return None

    config['feature extractor'] = check_statistics_config(experiment_name, tabulation, verbose)
    if config['feature extractor'] is None:
        return None

    config["session configuration"] = check_session_configuration(config["session configuration"], tabulation)

    if config["session configuration"] is None:
        return None

    return config


def check_test_config(experiment_name, test_name, tabulation='', verbose=True):

    configuration_rules = load_json(get_configuration_rules('test.json'))

    filename = get_test_name(experiment_name)

    if not check_existance('source', filename, True, tabulation, 0, verbose, True):
        return None

    tabulation = tabulation + ' ' + filename

    try:
        config = load_json(filename)
        if verbose:
            print tabulation, error_levels[3], 'File', filename, 'loaded'
    except:
        print tabulation, error_levels[0], "Error during loading", filename
        return None

    config = apply_config(configuration_rules, config, tabulation)

    config["data folder"] = get_data_name(test_name)
    config["output folder"] = get_tests_name(experiment_name, test_name)

    if not check_existance('data folder', config['data folder'], True, tabulation, 0, verbose, True, directory=True):
        return None

    config['classical'] = []
    config['testable'] = []
    if 'algorithms' not in config.keys():
        algorithms = {}
        multiplier = config['min size']
        step = config['step']
        classical_keys = []
        for alg in config['compare algorithms']:
            for size in config['check size']:
                class_info = name2class(alg)
                class_info['size'] = size
                possible_names = class2name(class_info)
                for name in possible_names:
                    algorithms[name] = None
                    classical_keys.append(name)
        config['classical'] = classical_keys
        for algorithm_type, experiment in config["algorithm type"]:
            for i in range(config['max size'] + 1):
                size = int(multiplier * (step ** i))
                class_info = name2class(algorithm_type)
                class_info['size'] = size
                possible_names = class2name(class_info)
                for name in possible_names:
                    algorithms[name] = experiment
                    config['testable'].append(name)
        config['algorithms'] = algorithms

    response = load_caching_algorithms(config['algorithms'], tabulation, verbose)
    if response is None:
        return None
    else:
        featurers, admission, eviction, models, common_models = response

    config['featurers'] = featurers
    config['admission'] = admission
    config['eviction'] = eviction
    config['models'] = models
    config['common models'] = common_models

    return config
