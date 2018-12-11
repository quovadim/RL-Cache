from environment_aux import load_json, name_to_class
import os.path

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
                print tabulation, error_levels[3], 'Field', name, 'correct with value', value
        if not is_exact:
            print tabulation, error_levels[2], 'Field', name, 'converted from', orignal_type, 'to', target_type
    else:
        print tabulation, error_levels[level], name, 'has wrong type', type(value)
    return is_exact, can_be_converted, target_type


def check_fiends(config, names, types, necessity, tabulation, verbose):
    for i in range(len(names)):
        if necessity:
            exact, converted, target = check_single_type(names[i], config[names[i]], types[i], verbose, tabulation, 0)
            if not exact and converted:
                config[names[i]] = target(config[names[i]])
            if not exact and not converted:
                return None
    return config


def check_range(name, value, interval_left, interval_right, tabulation, level, verbose, dead):
    condition_left = interval_left is not None and value < interval_left
    condition_right = interval_right is not None and value > interval_right
    if condition_left:
        if dead:
            print tabulation, error_levels[level], name, 'with value', value, 'is less than', value > interval_left
        return False

    if condition_right:
        if dead:
            print tabulation, error_levels[level], name, 'with value', value, 'is higher than', value > interval_left
        return False

    if verbose:
        if interval_left is None:
            interval_left = '-inf'
        if interval_right is None:
            interval_right = '+inf'
        print tabulation, error_levels[3], name, 'with value', value, \
            'is in interval from', interval_left, 'to', interval_right

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
                               verbose, True):
                return None
            if recommendations is not None:
                lf, rf = recommendations
                check_range(names[i], config[names[i]], lf, rf, tabulation, level + 1,
                            verbose, True)
    return config


def check_algorithm(name, value, random, tabulation, level, verbose, dead):
    is_exact, can_be_converted, target_type = check_single_type(name, value, [str, unicode], verbose, tabulation, level)

    if not is_exact and can_be_converted:
        value = target_type(value)

    if not is_exact and not can_be_converted:
        return False, name

    try:
        _, admission_random, _, eviction_random, _, _ = name_to_class(value)
        if verbose:
            print tabulation, error_levels[3], 'for', name, value, 'is correct algorithm name'
    except AssertionError:
        if dead:
            print tabulation, error_levels[level], 'for', name, value, 'is not correct algorithm name'
            return False, name

    if random == (get_randomness(value) == 0):
        reality = 'random' if not random else 'deterministic'
        theory = 'random' if random else 'deterministic'
        if dead:
            print tabulation, error_levels[level], 'for', name, value, 'should be', theory, 'while it is', reality
            return False, name
    else:
        reality = 'random' if random else 'deterministic'
        if verbose:
            print tabulation, error_levels[3], 'for', name, value, 'is', reality, 'as it should be'
    return True, value


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

    if verbose:
        tabulation = '\t' + tabulation
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

    if not check_existance('statistics', config['statistics'], True, tabulation, 0, verbose, True):
        return None

    num_lines = sum(1 for _ in open(config['statistics'], 'r'))
    intervals[names.index('warmup')] = (0, num_lines, None)

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
        (1, None, None),
        (0, None, None),
        (1, None, None),
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

    if verbose:
        tabulation = '\t' + tabulation
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
        (0, None, None),
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
        (0, None, None),
        (0, 100, None)
    ]

    types = [[str, unicode],
             [str, unicode],
             [int],
             [str, unicode],
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
                 False,
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

    if verbose:
        tabulation = '\t' + tabulation
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

    correct, config['target'] = check_algorithm('target', config['target'], True, tabulation, 0, verbose, True)
    if not correct:
        return None

    new_names = []
    for item in config['algorithms']:
        correct, alg = check_algorithm(item, item, False, tabulation, 0, verbose, True)
        if not correct:
            return None
        new_names.append(alg)
    config['algorithms'] = new_names

    config["session configuration"] = check_session_configuration(config["session configuration"], tabulation, verbose)

    if config["session configuration"] is None:
        return None

    return config
