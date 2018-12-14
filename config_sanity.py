from environment_aux import load_json, name_to_class
from FeatureExtractor import PacketFeaturer
from model import *
import os.path
from copy import deepcopy

import keras as K

import gc

error_levels = {
    0: '\033[1m\033[91mCRITICAL\033[0m',
    1: '\033[1m\033[93mWARNING\033[0m',
    2: '\033[1m\033[94mFIXED\033[0m',
    3: '\033[1m\033[92mOK\033[0m'
}


def check_single_type(name, value, type_list, verbose, tabulation, level):
    is_exact = False
    can_be_converted = False
    orignal_type = None
    target_type = None
    for type_class in type_list:
        if isinstance(value, type_class):
            is_exact = True
            break
        else:
            try:
                target_type = type_class
                orignal_type = type(value)
                can_be_converted = True
            except:
                pass
    if is_exact or can_be_converted:
        if verbose:
            if is_exact:
                print tabulation, error_levels[3], 'Field', name, 'type', type(value), 'is correct'
        if not is_exact:
            print tabulation, error_levels[2], 'Field', name, 'converted from', orignal_type, 'to', target_type
    else:
        print tabulation, error_levels[level], name, 'has wrong type', type(value)
    return is_exact, can_be_converted, target_type


def check_fiends(config, names, types, necessity, tabulation, verbose):
    for i in range(len(names)):
        try:
            value = config[names[i]]
        except KeyError:
            if necessity[i]:
                print tabulation, error_levels[0], names[i], 'does not exist'
                return None
            else:
                print tabulation, error_levels[1], names[i], 'does not exist'
                continue

        exact, converted, target = check_single_type(names[i], value, types[i], verbose, tabulation, 0)
        if not exact and converted:
            config[names[i]] = target(value)
        if not exact and not converted:
            return None
    return config


def check_range(name, value, interval_left, interval_right, tabulation, level, verbose, dead, recommendation):
    condition_left = interval_left is not None and value < interval_left
    condition_right = interval_right is not None and value > interval_right
    value_type = 'obligatory' if not recommendation else 'recommended'
    if condition_left:
        if dead:
            print tabulation, error_levels[level], name, '=', value, 'is less than', value_type, interval_left
        return False

    if condition_right:
        if dead:
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


def get_randomness(value):
    _, admission_random, _, eviction_random, _, _ = name_to_class(value)
    randomness = 0
    if admission_random:
        randomness += 1
    if eviction_random:
        randomness += 1
    return randomness


def check_ranges(config, names, necessity, intervals, verbose, tabulation, level):
    for i in range(len(names)):
        if necessity[i] and intervals[i] is not None:
            lf, rf, recommendations = intervals[i]
            if not check_range(names[i], config[names[i]], lf, rf, tabulation, level,
                               verbose, True, False):
                return None
            if recommendations is not None:
                lf, rf = recommendations
                check_range(names[i], config[names[i]], lf, rf, tabulation, level + 1,
                            verbose, True, True)
    return config


def check_algorithm(name, value, model, tabulation, level, verbose, dead):
    is_exact, can_be_converted, target_type = check_single_type(name, value, [str, unicode], verbose, tabulation, level)

    if not is_exact and can_be_converted:
        value = target_type(value)

    if not is_exact and not can_be_converted:
        return False, value, None

    if model is not None:
        is_exact, can_be_converted, target_type = check_single_type(
            value + ' model', model, [str, unicode], verbose, tabulation, level)

        if not is_exact and can_be_converted:
            model = target_type(model)

        if not is_exact and not can_be_converted:
            return False, value, None

    try:
        _, admission_random, _, eviction_random, _, _ = name_to_class(value)
        if verbose:
            print tabulation, error_levels[3], 'for', name, value, 'is correct algorithm name'
    except AssertionError:
        if dead:
            print tabulation, error_levels[level], 'for', name, value, 'is not correct algorithm name'
            return False, value, None

    randomness_value = get_randomness(value)

    if randomness_value == 0:
        if model is None:
            if verbose:
                print tabulation, error_levels[3], 'for', name, value, 'matching model'
            return True, value, model
        else:
            print tabulation, error_levels[level], 'for', name, value, 'model mismatch'
            return False, value, None

    if randomness_value != 0:
        if model is None:
            print tabulation, error_levels[1], 'for', name, value, 'assuming local model'
            return True, value, model
        else:
            if verbose:
                print tabulation, error_levels[3], 'for', name, value, 'matching model', model
            return True, value, check_model_config(model, tabulation, verbose)


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
    statistics = {}
    for key in algorithms.keys():
        model = algorithms[key]
        if model is not None:
            target_index = -1
            model, adm_path, evc_path = model
            model_config = check_model_config(model, tabulation=tabulation, verbose=verbose)
            if model_config is None:
                return None
            if model_config['statistics'] in known_names:
                target_index = known_names.index(model_config['statistics'])
            else:
                statistics_config = check_statistics_config(model_config['statistics'], verbose=False)
                if statistics_config is None:
                    return None
                for i in range(len(known_configs)):
                    if compare_statistics_dicts(statistics_config, known_configs[i]):
                        target_index = i
                        break
                if target_index < 0:
                    target_index = len(known_names)
                    known_names.append(model_config['statistics'])
                    known_configs.append(statistics_config)
                    featurers.append(PacketFeaturer(statistics_config))

            adm_model = 0
            evc_model = 0
            cm_model = None
            input_dim = featurers[target_index].dim
            if model_config['use common']:
                cm_model = create_common_model(model_config, input_dim)
            common_models_mapping[key] = cm_model
            class_type, rng_adm, adm_index, rng_evc, evc_index, rng_model = name_to_class(key)
            name_to_use = [class_type, rng_adm, adm_index, rng_evc, evc_index, adm_path, evc_path]
            name_to_use = '|'.join([str(item) for item in name_to_use])
            if rng_adm:
                if name_to_use in known_admission_names:
                    adm_model = known_admission_names.index(name_to_use)
                else:
                    adm_model = len(known_admission_models)
                    model = create_admission_model(model_config, input_dim, cm_model)
                    if adm_path is not None:
                        model.load_weights(adm_path)
                    known_admission_models.append(model)
                    known_admission_names.append(name_to_use)
            if rng_evc:
                if name_to_use in known_eviction_names:
                    evc_model = known_eviction_names.index(name_to_use)
                else:
                    evc_model = len(known_eviction_models)
                    model = create_eviction_model(model_config, input_dim, cm_model)
                    if evc_path is not None:
                        model.load_weights(evc_path)
                    known_eviction_models.append(model)
                    known_eviction_names.append(name_to_use)
            models[key] = (adm_model, evc_model)
            statistics[key] = target_index
        else:
            statistics[key] = 0
            models[key] = (0, 0)
    return featurers, statistics, known_admission_models, known_eviction_models, models, common_models_mapping


def check_statistics_config(filename, tabulation='', verbose=True):
    names = ["statistics",
             "warmup",
             "split step",
             "normalization limit",
             "bias",
             "save",
             "load",
             "filename",
             "show stat",
             "lambda"]

    intervals = [
        None,
        None,
        (0, None, None),
        (0, None, (1, None)),
        None,
        None,
        None,
        None,
        None,
        (0, 1, None)
    ]

    types = [[str, unicode],
             [int],
             [int],
             [float, int],
             [float, int],
             [bool],
             [bool],
             [str, unicode],
             [bool],
             [float]]

    necessity = [True,
                 True,
                 True,
                 True,
                 True,
                 True,
                 True,
                 True,
                 True,
                 True]

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

    config = check_fiends(config, names, types, necessity, tabulation, verbose)
    if config is None:
        return config

    # No reason to check it, we can live without this statistics
    #if not check_existance('statistics', config['statistics'], True, tabulation, 0, verbose, True):
    #    return None

    #num_lines = sum(1 for _ in open(config['statistics'], 'r'))
    #intervals[names.index('warmup')] = (0, num_lines, None)

    config = check_ranges(config, names, necessity, intervals, verbose, tabulation, 0)
    if config is None:
        return config

    if not config['save'] and not config['load']:
        print tabulation, error_levels[1], 'We would strongly recommend you to save or load the data'

    if not os.path.isfile(config['filename']) and config['load']:
        print tabulation, error_levels[1], 'File', config['filename'], 'does not exists, data will be generated'

    if os.path.isfile(config['filename']) and config['save'] and not config['load']:
        print tabulation, error_levels[1], 'File', config['filename'], 'will be overwritten'

    return config


def check_model_config(filename, tabulation='', verbose=True):
    names = ["wing size",
             "dropout rate",
             "use common",
             "multiplier common",
             "layers common",
             "multiplier each",
             "layers each",
             "use batch normalization",
             "eviction lr",
             "admission lr",
             "statistics"]

    intervals = [
        (0, None, None),
        (0, 1, (0, 0.5)),
        None,
        (3, None, None),
        (0, None, None),
        (3, None, None),
        (0, None, None),
        None,
        (0, 1, (1e-6, 1e-2)),
        (0, 1, (1e-6, 1e-2)),
        None
    ]

    types = [[int],
             [float],
             [bool],
             [int],
             [int],
             [int],
             [int],
             [bool],
             [float],
             [float],
             [str, unicode]]

    necessity = [True,
                 True,
                 True,
                 False,
                 False,
                 True,
                 True,
                 True,
                 True,
                 True,
                 True]

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

    config = check_fiends(config, names, types, necessity, tabulation, verbose)
    if config is None:
        return config

    if config["use common"]:
        necessity_common = [False] * len(names)
        necessity_common[names.index("multiplier common")] = True
        necessity_common[names.index("layers common")] = True

        necessity[names.index("multiplier common")] = True
        necessity[names.index("layers common")] = True

        config = check_fiends(config, names, types, necessity_common, tabulation, verbose)
        if config is None:
            return config

    if not check_existance('statistics', config['statistics'], True, tabulation, 0, verbose, True):
        return None

    config = check_ranges(config, names, necessity, intervals, verbose, tabulation, 0)

    return config


def check_session_configuration(config, tabulation, verbose):
    names = [
        "collect discounted",
        "change",
        "change mode",
        "seed",
        "initial gamma",
        "gamma",
        "alpha"
    ]

    intervals = [
        None,
        None,
        None,
        None,
        (0, 1, (0.1, 0.9)),
        (0, 1, (1e-10, 0.1)),
        (-1, 2, None)
    ]

    types = [
        [bool],
        [bool],
        [str, unicode],
        [int],
        [float],
        [float],
        [int]
    ]

    necessity = [
        True,
        True,
        False,
        False,
        False,
        False,
        True
    ]

    tabulation = tabulation + ' ' + 'session'

    config = check_fiends(config, names, types, necessity, tabulation, verbose)
    if config is None:
        return config

    if config["collect discounted"]:
        necessity_common = [False] * len(names)
        necessity_common[names.index("initial gamma")] = True
        necessity_common[names.index("gamma")] = True

        necessity[names.index("gamma")] = True
        necessity[names.index("initial gamma")] = True

        config = check_fiends(config, names, types, necessity_common, tabulation, verbose)
        if config is None:
            return config

    if config["change"]:
        necessity_common = [False] * len(names)
        necessity_common[names.index("change mode")] = True

        necessity[names.index("change mode")] = True

        config = check_fiends(config, names, types, necessity_common, tabulation, verbose)
        if config is None:
            return config

    if config["change"] and str(config["change mode"]) == 'random':
        necessity_common = [False] * len(names)
        necessity_common[names.index("seed")] = True

        necessity[names.index("seed")] = True

        config = check_fiends(config, names, types, necessity_common, tabulation, verbose)
        if config is None:
            return config

    config = check_ranges(config, names, necessity, intervals, verbose, tabulation, 0)
    if config is None:
        return config

    return config


def check_train_config(filename, tabulation='', verbose=True):
    names = [
        "model",
        "data folder",
        "cache size",
        "target",
        "batch size",
        "seed",
        "session configuration",
        "samples",
        "max samples",
        "percentile admission",
        "percentile eviction",
        "epochs",
        "warmup",
        "runs",
        "refresh period",
        "repetitions",
        "drop",
        "store period",
        "overlap",
        "period",
        "refresh policy",
        "refresh value",
        "algorithms",
        "models",
        "iterative",
        "start iteration",
        "IP:train admission",
        "IP:train eviction",
        "dump sessions",
        "dump limit",
        "dump percentile"
    ]

    intervals = [
        None,
        None,
        (1, None, (32, 102400)),
        None,
        (1, None, (32, 40960)),
        None,
        None,
        (0, None, (75, 750)),
        (0, None, None),
        (50, 100, None),
        (50, 100, None),
        (0, None, None),
        (-1, None, None),
        (0, None, None),
        (0, None, None),
        (0, None, None),
        None,
        (1, None, None),
        None,
        (0, None, (10000, 1000000)),
        None,
        (0, None, None),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        (0, None, None),
        (0, 100, None)
    ]

    types = [[str, unicode],
             [str, unicode],
             [int],
             [str, unicode],
             [int],
             [int],
             [dict],
             [int],
             [int],
             [int, float],
             [int, float],
             [int],
             [int],
             [int],
             [int],
             [int],
             [bool],
             [int],
             [int],
             [int],
             [str, unicode],
             [int],
             [list],
             [list],
             [bool],
             [str, unicode],
             [bool],
             [bool],
             [bool],
             [int],
             [int, float]]

    necessity = [True,
                 True,
                 True,
                 True,
                 True,
                 True,
                 True,
                 True,
                 True,
                 True,
                 True,
                 True,
                 True,
                 True,
                 True,
                 True,
                 True,
                 False,
                 True,
                 True,
                 True,
                 True,
                 True,
                 True,
                 True,
                 False,
                 True,
                 True,
                 True,
                 False,
                 False]

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

    config = check_fiends(config, names, types, necessity, tabulation, verbose)
    if config is None:
        return config

    if not config["drop"]:
        necessity_common = [False] * len(names)
        necessity_common[names.index("store period")] = True
        necessity[names.index("store period")] = True

        config = check_fiends(config, names, types, necessity_common, tabulation, verbose)
        if config is None:
            return config

    if config["iterative"]:
        necessity_common = [False] * len(names)
        necessity_common[names.index("iterative")] = True
        necessity[names.index("iterative")] = True

        config = check_fiends(config, names, types, necessity_common, tabulation, verbose)
        if config is None:
            return config

    if config["dump sessions"]:
        necessity_common = [False] * len(names)
        necessity_common[names.index("dump limit")] = True
        necessity_common[names.index("dump percentile")] = True

        necessity[names.index("dump limit")] = True
        necessity[names.index("dump percentile")] = True

        config = check_fiends(config, names, types, necessity_common, tabulation, verbose)
        if config is None:
            return config

    intervals[names.index("overlap")] = (-1, config['period'], None)

    if not check_existance('model', config['model'], True, tabulation, 0, verbose, True):
        return None

    if not check_existance('data folder', config['data folder'], True, tabulation, 0, verbose, True, directory=True):
        return None

    config = check_ranges(config, names, necessity, intervals, verbose, tabulation, 0)
    if config is None:
        return None

    correct, config['target'], local_model = check_algorithm(
        'target', config['target'], config['model'], tabulation, 0, verbose, True)
    if not correct:
        return None

    config['model'] = local_model
    config['feature extractor'] = check_statistics_config(local_model['statistics'], tabulation, verbose)
    if config['feature extractor'] is None:
        return None

    new_names = []
    models = []
    extractors = []
    for i in range(len(config['algorithms'])):
        algorithm_name = config['algorithms'][i]
        model_name = config['models'][i]
        correct, algorithm_name, model = \
            check_algorithm(algorithm_name, algorithm_name, model_name, tabulation, 0, verbose, True)
        if not correct:
            return None
        if model is None:
            model = config['model']
            extractor = config['feature extractor']
        else:
            extractor = check_statistics_config(model, tabulation, verbose)
        if extractor is None:
            return None
        new_names.append(algorithm_name)
        models.append(model)
        extractors.append(extractor)

    config['models'] = models
    config['algorithms'] = new_names
    config['extractors'] = extractors

    config["session configuration"] = check_session_configuration(config["session configuration"], tabulation, verbose)

    if config["session configuration"] is None:
        return None

    return config


def check_test_config(filename, tabulation='', verbose=True):
    names = [
        "data folder",
        "cache size",
        "batch size",
        "seed",
        "period",
        "algorithms",
        "reset",
        "alpha",
        "warmup"
    ]

    intervals = [
        None,
        (1, None, (32, 102400)),
        (1, None, (32, 40960)),
        None,
        (0, None, (600, 3600)),
        None,
        None,
        None,
        (0, None, (50000, None))
    ]

    types = [[str, unicode],
             [int],
             [int],
             [int],
             [int],
             [dict],
             [bool],
             [int],
             [int]]

    necessity = [True,
                 True,
                 True,
                 True,
                 True,
                 True,
                 True,
                 True,
                 True,
                 True]

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

    config = check_fiends(config, names, types, necessity, tabulation, verbose)
    if config is None:
        return config

    if not check_existance('data folder', config['data folder'], True, tabulation, 0, verbose, True, directory=True):
        return None

    config = check_ranges(config, names, necessity, intervals, verbose, tabulation, 0)
    if config is None:
        return None

    response = load_caching_algorithms(config['algorithms'], tabulation, verbose)
    if response is None:
        return None
    else:
        featurers, statistics, admission, eviction, models, common_models = response

    config['featurers'] = featurers
    config['statistics'] = statistics
    config['admission'] = admission
    config['eviction'] = eviction
    config['models'] = models
    config['common models'] = common_models

    return config
