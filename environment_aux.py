import numpy as np
from MLSim import MLSimulator
from GDSim import GDSimulator
from LRUSim import LRUSimulator

import sys
import pickle
from multiprocessing import Process, Queue
from copy import deepcopy
import random


def to_ts(time_diff):
    seconds = time_diff % 60
    time_diff /= 60
    minutes = time_diff % 60
    time_diff /= 60
    hours = time_diff % 60
    time_diff = '{:2d}:{:2d}:{:2d}'.format(hours, minutes, seconds)
    return time_diff.replace(' ', '0')


def threaded(f, daemon=False):

    def wrapped_f(q, *args, **kwargs):
        ret = f(*args, **kwargs)
        q.put(ret)

    def wrap(*args, **kwargs):
        q = Queue()

        t = Process(target=wrapped_f, args=(q,)+args, kwargs=kwargs)
        t.daemon = daemon
        t.start()
        t.result_queue = q
        return t

    return wrap


def copy_object(object_to_copy):
    res = deepcopy(object_to_copy)
    res.set_ratings(object_to_copy.get_ratings())

    return res


def name_to_class(name):
    name = name.split('-')
    name_admission = name[0]
    name_eviction = name[1]
    if len(name) > 2:
        name_rng = name[2]
    else:
        name_rng = None
    admission_random = False
    eviction_random = False
    class_type = None
    admission_index = -1
    eviction_index = -1
    randomness_type = None
    if name_rng is not None:
        if name_rng == 'RNG':
            randomness_type = True
        if name_rng == 'DET':
            randomness_type = False
    if name_admission == 'AL':
        admission_random = False
        admission_index = -1
    if name_admission == 'SH':
        admission_random = False
        admission_index = 2
    if name_admission == 'ML':
        admission_random = True
        admission_index = -1
    if name_eviction == 'GDSF':
        class_type = GDSimulator
        eviction_random = False
        eviction_index = 0
    if name_eviction == 'LRU':
        class_type = LRUSimulator
        eviction_random = False
        eviction_index = 1
    if name_eviction == 'LFU':
        class_type = LRUSimulator
        eviction_random = False
        eviction_index = 4
    if name_eviction == 'Oracle':
        class_type = LRUSimulator
        eviction_random = False
        eviction_index = 3
    if name_eviction == 'ML':
        class_type = MLSimulator
        eviction_random = True
        eviction_index = -1
    if name_eviction == 'MLGD':
        class_type = GDSimulator
        eviction_random = True
        eviction_index = -1
    if name_eviction == 'MLLRU':
        class_type = LRUSimulator
        eviction_random = True
        eviction_index = -1

    assert class_type is not None

    return class_type, admission_random, admission_index, eviction_random, eviction_index, randomness_type


def metric_funct(x, y, alpha):
    return x * alpha + y * (1 - alpha)


def metric(algorithm, alpha):
    return metric_funct(algorithm.byte_hit_rate(), algorithm.hit_rate(), alpha)


def compute_rating(value, total):
    w = (total - 1) / 2
    return 2. ** (float(value - w))


def sampling(prob_matrix):
    maximum_allowed = prob_matrix.shape[1]
    prob_matrix = prob_matrix.T
    s = prob_matrix.cumsum(axis=0)
    r = np.random.rand(prob_matrix.shape[1])
    k = (s < r).sum(axis=0)
    return [min(item, maximum_allowed - 1) for item in k]


def get_unique_dict(data, labels=None):
    unique_data, data_counts = np.unique(data, return_counts=True)
    evc_counts = data_counts * 100. / len(data)
    data_combined = []
    collected_labels = set()
    for i in range(len(unique_data)):
        data_combined.append((unique_data[i], evc_counts[i]))
        collected_labels.add(unique_data[i])
    for item in labels:
        if item not in collected_labels:
            data_combined.append((item, 0))
    data_combined = sorted(data_combined, key=lambda x: x[0])
    result_data = []
    for item in data_combined:
        result_data.append(item[0])
        result_data.append(item[1])
    return result_data


def select_elites(states, actions, rewards, percentile, max_samples):
    percentile_value = percentile
    if percentile is not None:
        elite_indicies = [i for i in range(len(rewards)) if rewards[i] >= percentile]
    else:
        elite_indicies = [np.argmax(rewards)]
        percentile_value = max(rewards)
    if len(elite_indicies) > max_samples:
        selite = [(item, rewards[item]) for item in elite_indicies]
        selite = sorted(selite, key=lambda x: x[0], reverse=True)
        elite_indicies = [item[0] for item in selite[:max_samples]]
        percentile_value = selite[max_samples - 1][1]
    elite_states = np.concatenate([states[i] for i in elite_indicies], axis=0)
    elite_actions = np.concatenate([actions[i] for i in elite_indicies], axis=0)
    return elite_states, elite_actions, percentile_value


def generate_data_for_models(keys,
                             keys_info,
                             classical_feature_set,
                             ml_feature_set,
                             adm_model,
                             evc_model,
                             batch_size):
    decisions_adm = {}
    decisions_evc = {}
    predictions_evc = None
    predictions_adm = None
    amax_adm = None
    amax_evc = None
    classical_admission = [True] * len(classical_feature_set)

    for key in keys:
        class_type, rng_adm, adm_index, rng_evc, evc_index, eng_type = keys_info[key]
        if rng_adm:
            if predictions_adm is None:
                predictions_adm = adm_model.predict(ml_feature_set, verbose=0, batch_size=batch_size)
                amax_adm = [bool(np.argmax(item) == 1) for item in predictions_adm]
            assert eng_type is not None
            if eng_type:
                decisions_adm[key] = [bool(item == 1) for item in sampling(predictions_adm)]
            else:
                decisions_adm[key] = amax_adm
        else:
            if adm_index < 0:
                decisions_adm[key] = classical_admission
            else:
                decisions_adm[key] = [bool(item == 1) for item in classical_feature_set[:, adm_index]]

        if rng_evc:
            if predictions_evc is None:
                predictions_evc = evc_model.predict(ml_feature_set, verbose=0, batch_size=batch_size)
                amax_evc = [compute_rating(np.argmax(item), predictions_evc.shape[1])
                            for item in predictions_evc]
            assert eng_type is not None
            if eng_type:
                decisions_evc[key] = [compute_rating(item, predictions_evc.shape[1])
                                      for item in sampling(predictions_evc)]
            else:
                decisions_evc[key] = amax_evc
        else:
            assert evc_index >= 0
            decisions_evc[key] = classical_feature_set[:, evc_index]
    return decisions_adm, decisions_evc


def train_model(percentile,
                model,
                rewards,
                states,
                actions,
                predictions,
                features_embedding,
                answers_embedding,
                epochs,
                batch_size,
                label):
    if percentile is not None:
        percentile = np.percentile(rewards, percentile)
    elite_states, elite_actions, percentile_value = select_elites(states, actions, rewards, percentile)

    data = sampling(predictions)
    unique_sampled = get_unique_dict(data, range(predictions.shape[1]))
    unique_sampled = [int(unique_sampled[i]) if i % 2 == 0 else unique_sampled[i]
                          for i in range(len(unique_sampled))]
    rstr_random = ' '.join(['{:4d} : {:7.3f}%'
                            for _ in range(len(unique_sampled) / 2)]).format(*unique_sampled)

    print label, rstr_random

    v = model.fit(features_embedding[elite_states], answers_embedding[elite_actions],
                  epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0)

    samples = [i for i in range(len(rewards)) if rewards[i] >= percentile]

    print 'Samples: {:6d} Accuracy: {:7.4f}% Loss: {:7.5f} ' \
          'Mean: {:7.4f}% Median: {:7.4f}% Percentile: {:7.4f}% Max: {:7.4f}%'.format(
        len(samples),
        100 * np.mean(v.history['acc'][0]),
        np.mean(v.history['loss'][0]),
        100 * np.mean(rewards),
        100 * np.median(rewards),
        100 * percentile_value,
        100 * max(rewards))


def test_algorithms(algorithms,
                    predictions_adm,
                    predictions_evc,
                    rows,
                    alpha,
                    output_file=None,
                    base_iteration=0,
                    special_keys=[]):

    counter = 0

    total_size = sum([row['size'] for row in rows])
    total_time = rows[len(rows) - 1]['timestamp'] - rows[0]['timestamp']

    history = {'time': [], 'flow': 0}
    keys = sorted(algorithms.keys())
    for key in keys:
        history[key] = []

    for row in rows:
        for alg in keys:
            algorithms[alg].decide(row, predictions_evc[alg][counter], predictions_adm[alg][counter])

        if output_file is not None and (counter % 100 < 0 or counter == len(rows) - 1):
            history['flow'] = float(total_size) / (1e-4 + float(total_time))
            history['time'].append(row['timestamp'])
            for key in keys:
                history[key].append(metric(algorithms[key], alpha))

        if counter % 100 == 0 or counter == len(rows) - 1:
            names = keys
            values = [100 * metric(algorithms[alg], alpha) for alg in keys]
            best_performance = keys[values.index(max(values))]
            print_list = []
            for name in names:
                if name in special_keys:
                    if name == best_performance:
                        print_list.append('\033[93m{:^' + str(len(name)) + 's}\033[0m \033[1m{:5.2f}%\033[0m')
                    else:
                        print_list.append('\033[92m{:^' + str(len(name)) + 's}\033[0m \033[1m{:5.2f}%\033[0m')
                else:
                    if name == best_performance:
                        print_list.append('\033[93m{:^' + str(len(name)) + 's}\033[0m \033[1m{:5.2f}%\033[0m')
                    else:
                        print_list.append('{:^' + str(len(name)) + 's} \033[1m{:5.2f}%\033[0m')
            print_string = ' | '.join(print_list)
            print_string = 'Iteration {:10d} ' + print_string
            subst_vals = []
            for i in range(len(names)):
                subst_vals.append(names[i])
                subst_vals.append(values[i])
            sys.stdout.write('\r' + print_string.format(*([base_iteration + counter + 1] + subst_vals)))
            sys.stdout.flush()

        counter += 1

    print ''

    if output_file is not None:
        pickle.dump(history, open(output_file, 'w'))


def get_session_features(bool_eviction, bool_admission, predictions_evc, predictions_adm):
    eviction_defined, eviction_deterministic = bool_eviction
    admission_defined, admission_deterministic = bool_admission
    if not eviction_defined:
        if eviction_deterministic:
            eviction_decisions = np.argmax(predictions_evc, axis=1)
        else:
            eviction_decisions = sampling(predictions_evc)
        evc_decision_values = eviction_decisions
        eviction_decisions = [compute_rating(eviction_decisions[i], len(predictions_evc[i]))
                              for i in range(len(eviction_decisions))]
    else:
        evc_decision_values = None
        eviction_decisions = predictions_evc

    if not admission_defined:
        if admission_deterministic:
            admission_decisions = np.argmax(predictions_adm, axis=1)
        else:
            admission_decisions = sampling(predictions_adm)
        adm_decision_values = admission_decisions
        admission_decisions = [bool(admission_decisions[i] == 1) for i in range(len(eviction_decisions))]
    else:
        adm_decision_values = None
        admission_decisions = predictions_adm
    return evc_decision_values, eviction_decisions, adm_decision_values, admission_decisions


def generate_session_continious(
        predictions_evc,
        predictions_adm,
        rows,
        algorithm_template,
        config,
        eviction_deterministic=False,
        admission_deterministic=False,
        eviction_defined=False,
        admission_defined=False,
        collect_eviction=True,
        collect_admission=True):

    lstates = []
    lactions = []
    lstates_adm = []
    lactions_adm = []

    np.random.seed()

    algorithm = copy_object(algorithm_template)

    algorithm.reset()

    alpha = config['alpha']

    ratings = algorithm.get_ratings()

    if config['collect discounted']:
        reward_hits = 0
        reward_total = 0
        gamma = config['gamma'] ** (1.0/len(rows))
        multiplier = 1

    evc_decision_values, eviction_decisions, adm_decision_values, admission_decisions = \
        get_session_features((eviction_defined, eviction_deterministic),
                             (admission_defined, admission_deterministic),
                             predictions_evc,
                             predictions_adm)

    exchanged = False

    for i in range(len(rows)):

        hit = algorithm.decide(rows[i], eviction_decisions[i], admission_decisions[i])
        if config['collect discounted']:
            if hit:
                reward_hits += multiplier * metric_funct(rows[i]['size'], 1, alpha)
            reward_total += multiplier * metric_funct(rows[i]['size'], 1, alpha)
            multiplier *= gamma

        if exchanged:
            continue

        if config['randomness change'] and i > config['point of change']:
            eviction_deterministic = True
            admission_deterministic = True
            _, eviction_decisions, _, admission_decisions = \
                get_session_features((eviction_defined, eviction_deterministic),
                                     (admission_defined, admission_deterministic),
                                     predictions_evc,
                                     predictions_adm)
            exchanged = True
            continue

        if config['seeded change'] and i > config['point of change']:
            np.random.seed(config['seed'])
            random.seed(config['seed'])
            _, eviction_decisions, _, admission_decisions = \
                get_session_features((eviction_defined, eviction_deterministic),
                                     (admission_defined, admission_deterministic),
                                     predictions_evc,
                                     predictions_adm)
            exchanged = True
            continue

        if collect_eviction and not eviction_deterministic and not eviction_defined \
                and algorithm.prediction_updated_eviction:
            lstates.append(i)
            lactions.append(evc_decision_values[i])
        if collect_admission and not admission_deterministic and not admission_defined \
                and algorithm.prediction_updated_admission:
            lstates_adm.append(i)
            lactions_adm.append(adm_decision_values[i])

    if collect_admission:
        admission_rating = metric(algorithm, alpha)
    else:
        admission_rating = 0

    if collect_eviction:
        eviction_rating = metric(algorithm, alpha)
    else:
        eviction_rating = 0

    if config['collect discounted']:
        eviction_rating = reward_hits / reward_total
        admission_rating = reward_hits / reward_total

    assert algorithm_template.get_ratings() == ratings

    return lstates, np.asarray(lactions), lstates_adm, np.asarray(lactions_adm), eviction_rating, admission_rating


def gen_feature_set(rows, featurer, forget_lambda, memory_vector=None, classical=False, pure=False):
    feature_matrix = []
    featurer.reset()
    counter = 0
    memory_features = []
    if memory_vector is None and not classical:
        memory_vector = np.zeros(featurer.dim)

    for row in rows:
        featurer.update_packet_state(row)
        if classical:
            if pure:
                data = featurer.get_packet_features_pure(row)
            else:
                data = featurer.get_packet_features_classical(row)
        else:
            data = featurer.get_packet_features(row)
        feature_matrix.append(data)
        featurer.update_packet_info(row)
        counter += 1

    if not classical:
        for item in feature_matrix:
            memory_features.append(memory_vector)
            memory_vector = memory_vector * forget_lambda + item * (1 - forget_lambda)
        memory_features = np.asarray(memory_features)
        feature_matrix = np.concatenate([feature_matrix, memory_features], axis=1)
        return feature_matrix, memory_vector

    return np.asarray(feature_matrix)


def iterate_dataset(filenames):
    for fname in filenames:
        names = ['timestamp', 'id', 'size', 'frequency', 'lasp_app', 'log_time',
                'exp_recency', 'exp_log']#, 'future']
        types = [int, int, int, int, int, int, float, float]#, float]
        hdlr = open(fname, 'r')

        for line in hdlr:
            lines_converted = line.split(' ')
            lines_converted = [types[i](lines_converted[i]) for i in range(len(types))]
            yield dict(zip(names, lines_converted))

        hdlr.close()


def split_feature(feature, perc_steps):
    percs = [i * 100 / perc_steps for i in range(perc_steps + 1)]
    percentiles = [np.percentile(feature, item) for item in percs]
    percentiles[0] -= 1
    percentiles[len(percentiles) - 1] += 1
    percentiles = list(np.unique(percentiles))
    percentiles = sorted(percentiles)
    return [(percentiles[i-1], percentiles[i]) for i in range(1, len(percentiles))]
