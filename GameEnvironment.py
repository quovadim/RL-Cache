from os import listdir
from os.path import isfile, join
import sys
import numpy as np
from hurry.filesize import size as fsize
from tqdm import tqdm
import keras.layers as l
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from MLSim import MLSimulator
from GDSim import GDSimulator
from LRUSim import LRUSimulator
from FeatureExtractor import PacketFeaturer
import json

from multiprocessing import Process, Queue
from copy import deepcopy

import pickle


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


alpha = 1


def metric_funct(x, y):
    return x * alpha + y * (1 - alpha)


def metric(algorithm):
    return metric_funct(algorithm.byte_hit_rate(), algorithm.hit_rate())


def compute_rating(value, total):
    w = (total - 1) / 2
    return 2. ** (float(value - w))


def vectorized(prob_matrix):
    prob_matrix = prob_matrix.T
    s = prob_matrix.cumsum(axis=0)
    r = np.random.rand(prob_matrix.shape[1])
    k = (s < r).sum(axis=0)
    return k


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


def generate_session_continious(
        predictions_evc,
        predictions_adm,
        rows,
        algorithm_template,
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

    reward_hits = 0
    reward_total = 0
    gamma = 0.01 ** (1.0/len(rows))
    multiplier = 1

    if not eviction_defined:
        if eviction_deterministic:
            eviction_decisions = np.argmax(predictions_evc, axis=1)
        else:
            eviction_decisions = vectorized(predictions_evc)
        evc_decision_values = eviction_decisions
        eviction_decisions = [compute_rating(eviction_decisions[i], len(predictions_evc[i]))
                              for i in range(len(eviction_decisions))]
    else:
        eviction_decisions = predictions_evc

    if not admission_defined:
        if admission_deterministic:
            admission_decisions = np.argmax(predictions_adm, axis=1)
        else:
            admission_decisions = vectorized(predictions_adm)
        adm_decision_values = admission_decisions
        admission_decisions = [bool(admission_decisions[i] == 1) for i in range(len(eviction_decisions))]
    else:
        admission_decisions = predictions_adm

    for i in range(len(rows)):

        hit = algorithm.decide(rows[i], eviction_decisions[i], admission_decisions[i])
        #if hit:
        #    reward_hits += multiplier * metric_funct(rows[i]['size'], 1)
        #reward_total += multiplier * metric_funct(rows[i]['size'], 1)
        #multiplier *= gamma

        #if i > len(rows) / 2:
        #    eviction_deterministic = True
        #    admission_deterministic = True
        #    algorithm.deterministic_eviction = eviction_deterministic
         #   algorithm.deterministic_admission = admission_deterministic

        if collect_eviction and not eviction_deterministic and not eviction_defined \
                and algorithm.prediction_updated_eviction:
            lstates.append(i)
            lactions.append(evc_decision_values[i])
        if collect_admission and not admission_deterministic and not admission_defined \
                and algorithm.prediction_updated_admission:
            lstates_adm.append(i)
            lactions_adm.append(adm_decision_values[i])

    if collect_admission:
        admission_rating = metric(algorithm)
    else:
        admission_rating = 0

    if collect_eviction:
        eviction_rating = metric(algorithm)
    else:
        eviction_rating = 0

    #eviction_rating = reward_hits / reward_total
    #admission_rating = reward_hits / reward_total

    return lstates, np.asarray(lactions), lstates_adm, np.asarray(lactions_adm), eviction_rating, admission_rating


class GameEnvironment:
    def __init__(self, filepath, cache_size, ofname=None, verbose=False, save_markup=False, skip=0):
        self.filepath = filepath
        self.featurer = PacketFeaturer(load_name='auxiliary/features_primary.npy')
        self.verbose = verbose
        self.save_markup = save_markup
        self.cache_size = cache_size
        self.ofname = ofname
        self.skip = skip

        filenames = [(join(filepath, f), int(f.replace('.csv', ''))) for f in listdir(filepath)
                     if isfile(join(filepath, f)) and '.csv' in f and 'lock' not in f]
        filenames = sorted(filenames, key=lambda x: x[1])
        filenames = [item[0] for item in filenames]
        self.filenames = filenames
        self.current_file = self.skip
        self.frame = None
        self.data = None

        self.wing_size = 3
        self.last_dim = self.wing_size * 2 + 1

        dropout_rate = 0.0

        multiplier_common = 20
        multiplier_each = 100
        layers_common = 3
        layers_each = 3

        self.input_seq = 3

        self.cm = common_model = Sequential()
        common_model.add(l.Dense(self.featurer.dim * multiplier_common, input_shape=(2 * self.featurer.dim,),
                                 activation='elu'))
        common_model.add(l.BatchNormalization())
        for _ in range(layers_common):
            common_model.add(l.Dropout(dropout_rate))
            common_model.add(l.Dense(self.featurer.dim * multiplier_common, activation='elu'))

        self.model = Sequential()
        self.model.add(l.Dense(self.featurer.dim * multiplier_each, input_shape=(2 * self.featurer.dim,),
                               activation='elu'))
        #self.model.add(common_model)
        self.model.add(l.BatchNormalization())

        for i in range(layers_each):
            self.model.add(l.Dropout(dropout_rate))
            self.model.add(l.Dense(self.featurer.dim * int(multiplier_each * (layers_each - i) / layers_each),
                                   activation='elu'))
        self.model.add(l.Dropout(dropout_rate))
        self.model.add(l.Dense(self.last_dim, activation='softmax'))

        self.evc_optimizer = Adam(lr=1e-5)

        self.model.compile(self.evc_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        self.model.summary()

        self.model_admission = Sequential()

        self.model_admission.add(l.Dense(self.featurer.dim * multiplier_each, input_shape=(2 * self.featurer.dim,),
                                         activation='elu'))
        self.model_admission.add(l.BatchNormalization())
        #self.model_admission.add(common_model)
        for i in range(layers_each):
            self.model_admission.add(l.Dropout(dropout_rate))
            self.model_admission.add(l.Dense(self.featurer.dim * int(multiplier_each * (layers_each - i) / layers_each),
                                             activation='elu'))
        self.model_admission.add(l.Dropout(dropout_rate))
        self.model_admission.add(l.Dense(2, activation='softmax'))

        self.adm_optimizezr = Adam(lr=1e-5)

        self.model_admission.compile(self.adm_optimizezr, loss='binary_crossentropy', metrics=['accuracy'])

        self.model_admission.summary()

    @staticmethod
    def collect_features(ofname, t_max, filenames, skip, pure=None, verbose=True):
        feature_matrix = []
        counter = 0

        featurer = PacketFeaturer(pure)

        if pure is not None:
            ofile = open(ofname, 'w')

        try:
            for row in GameEnvironment.iterate_dataset_over_all(filenames, skip):
                if pure is None and counter % 500000 == 0:
                    np.save(ofname, feature_matrix)
                if counter > t_max:
                    break
                featurer.update_packet_state(row)
                if pure is None:
                    data = featurer.get_packet_features_pure(row).tolist()
                else:
                    data = featurer.get_packet_features(row).tolist()
                data.append(row['timestamp'])
                data.append(row['id'])
                data.append(row['size'])
                feature_matrix.append(np.asarray(data))
                if verbose:
                    if counter % 5000 == 0:
                        d = featurer.dim
                        if pure:
                            d = featurer.fnum
                        str_formatted = ' '.join(['{:^7.4f}' for _ in range(d)])
                        str_formatted = '{:10d}  ' + str_formatted
                        means = np.mean(feature_matrix, axis=0).tolist()
                        str_formatted = str_formatted.format(*([counter] + means))
                        sys.stdout.write('\r' + str_formatted)
                        sys.stdout.flush()
                if pure is not None and counter % 50000 == 0:
                    for line in feature_matrix:
                        ofile.write(' '.join([str(item) for item in line]) + '\n')
                    feature_matrix = []
                featurer.update_packet_info(row)
                counter += 1
            if verbose:
                print ''
        except KeyboardInterrupt:
            if pure is None:
                np.save(ofname, feature_matrix)
            else:
                ofile.close()

    def gen_feature_set(self, rows, c_vect=None, classical=False, pure=False, verbose=False):
        feature_matrix = []
        self.featurer.reset()
        counter = 0
        forget_lambda = 0.99
        memory_features = []
        if c_vect is None and not classical:
            c_vect = np.zeros(self.featurer.dim)
        if verbose:
            rows = tqdm(rows)
        for row in rows:
            self.featurer.update_packet_state(row)
            if classical:
                if pure:
                    data = self.featurer.get_packet_features_pure(row)
                else:
                    data = self.featurer.get_packet_features_classical(row)
            else:
                data = self.featurer.get_packet_features(row)
            feature_matrix.append(data)
            self.featurer.update_packet_info(row)
            counter += 1
        if not classical:
            for item in feature_matrix:
                memory_features.append(c_vect)
                c_vect = c_vect * forget_lambda + item * (1 - forget_lambda)
            memory_features = np.asarray(memory_features)
            feature_matrix = np.concatenate([feature_matrix, memory_features], axis=1)
            return feature_matrix, c_vect
        return np.asarray(feature_matrix)

    @staticmethod
    def test_algorithms(algorithms_ml,
                        algorithms_classic,
                        predictions_ml_adm,
                        predictions_ml_evc,
                        predictions_classic_adm,
                        predictions_classic_evc,
                        rows,
                        ofile=None,
                        verbose=True,
                        comparison_keys=None,
                        base_iteration=0):

        counter = 0

        total_size = sum([row['size'] for row in rows])
        total_time = rows[len(rows) - 1]['timestamp'] - rows[0]['timestamp']

        history = {'time': [], 'size': 0}
        for key in algorithms_classic.keys():
            history[key] = []
        for key in algorithms_ml.keys():
            history[key] = []

        for row in rows:
            for alg in algorithms_classic.keys():
                algorithms_classic[alg].decide(row,
                                               predictions_classic_evc[alg][counter],
                                               predictions_classic_adm[alg][counter])
            for alg in algorithms_ml.keys():
                algorithms_ml[alg].decide(row, predictions_ml_evc[alg][counter], predictions_ml_adm[alg][counter])

            if ofile is not None and (counter % 100 < 0 or counter == len(rows) - 1):
                history['flow'] = float(total_size) / (1e-4 + float(total_time))
                history['time'].append(row['timestamp'])
                for key in algorithms_classic.keys():
                    history[key].append(metric(algorithms_classic[key]))
                for key in algorithms_ml.keys():
                    history[key].append(metric(algorithms_ml[key]))

            if verbose:
                if counter % 100 == 0 or counter == len(rows) - 1:
                    names = algorithms_ml.keys() + algorithms_classic.keys()
                    values = [100 * metric(alg) for alg in algorithms_ml.values()] + \
                             [100 * metric(alg) for alg in algorithms_classic.values()]
                    spaces = [fsize(alg.free_space()) for alg in algorithms_ml.values()] + \
                             [fsize(alg.free_space()) for alg in algorithms_classic.values()]
                    print_string = ' | '.join(['{:^6s} HR {:5.2f}% FS {:^6s}' for _ in range(len(names))])
                    print_string = 'Iteration {:10d} ' + print_string
                    subst_vals = []
                    for i in range(len(names)):
                        subst_vals.append(names[i])
                        subst_vals.append(values[i])
                        subst_vals.append(spaces[i])
                    sys.stdout.write('\r' + print_string.format(*([base_iteration + counter + 1] + subst_vals)))
                    sys.stdout.flush()

            counter += 1

        if verbose:
            print
        if ofile is not None:
            pickle.dump(history, open(ofile, 'w'))
            if comparison_keys is not None:
                result = {}
                for key in comparison_keys:
                    result[key] = np.mean(history[key][len(history[key]) - 1])
                return result

    def test_ml_infinite(self,
                         t_max,
                         period,
                         filenames,
                         skip,
                         o_file_generator,
                         verbose=True):
        counter = 0
        current_rows = []
        file_counter = 0

        classes_ml = {'ML': MLSimulator, 'DET ML': MLSimulator}
        classes_classic = {'GDSF': GDSimulator, 'LRU': LRUSimulator, 'Oracle': LRUSimulator}

        algorithms_classic = {}
        for key in classes_classic:
            algorithms_classic[key] = classes_classic[key](self.cache_size)

        algorithms_ml = {}
        for key in classes_ml:
            algorithms_ml[key] = classes_ml[key](self.cache_size)

        c_vect = None

        start_time = None

        for row in GameEnvironment.iterate_dataset_over_all(filenames, skip):
            if counter > t_max:
                break

            if start_time is None:
                start_time = row['timestamp']

            if counter != 0 and row['timestamp'] - start_time > period:
                start_time = row['timestamp']
                print ''

                feature_set_classical = self.gen_feature_set(current_rows, classical=True, verbose=True)

                feature_set, c_vect = self.gen_feature_set(current_rows, classical=False, c_vect=c_vect, verbose=True)

                self.featurer.preserve()

                classical_admission = [True] * len(feature_set_classical)

                classical_features = [item.tolist() for item in feature_set_classical]

                predictions_classic_adm = {'GDSF': classical_admission,
                                           'LRU': classical_admission,
                                           'Oracle': classical_admission}

                predictions_classic_evc = {'GDSF': classical_features[:, 0],
                                           'LRU': classical_features[:, 1],
                                           'Oracle': classical_features[:, 3]}

                eviction_predictions = self.model.predict(feature_set, verbose=1, batch_size=4 * 4096)

                predictions_evc = {
                    'ML': [compute_rating(item, eviction_predictions.shape[1])
                           for item in vectorized(eviction_predictions)],
                    'DET ML': [compute_rating(np.argmax(item), eviction_predictions.shape[1])
                               for item in eviction_predictions]
                }

                admission_predictions = self.model_admission.predict(feature_set, verbose=1, batch_size=4 * 4096)

                predictions_adm = {
                    'ML': [item == 1
                           for item in vectorized(admission_predictions)],
                    'DET ML': [np.argmax(item) == 1 for item in admission_predictions]
                }

                for alg in algorithms_ml.keys():
                    algorithms_ml[alg].reset()
                for alg in algorithms_classic.keys():
                    algorithms_classic[alg].reset()

                GameEnvironment.test_algorithms(algorithms_ml,
                                                algorithms_classic,
                                                predictions_adm,
                                                predictions_evc,
                                                predictions_classic_adm,
                                                predictions_classic_evc,
                                                current_rows,
                                                o_file_generator + '_' + str(file_counter),
                                                verbose,
                                                base_iteration=counter - period)
                file_counter += 1
                current_rows = [row]
            else:
                current_rows.append(row)
                if verbose:
                    sys.stdout.write('\r' + 'Collector iteration {:7d}'.format(counter + 1))
                    sys.stdout.flush()
            counter += 1

    def test_adm_infinite(self,
                          t_max,
                          period,
                          filenames,
                          skip,
                          o_file_generator,
                          verbose=True):
        counter = 0
        current_rows = []
        file_counter = 0

        classes_ml = {'ML': MLSimulator, 'DET ML': MLSimulator}
        classes_classic = {'GDSF': GDSimulator, 'LRU': LRUSimulator, 'Oracle': LRUSimulator}

        algorithms_classic = {}
        for key in classes_classic:
            algorithms_classic[key] = classes_classic[key](self.cache_size)

        algorithms_ml = {}
        for key in classes_ml:
            algorithms_ml[key] = classes_ml[key](self.cache_size)

        c_vect = None

        start_time = None

        for row in GameEnvironment.iterate_dataset_over_all(filenames, skip):
            if counter > t_max:
                break

            if start_time is None:
                start_time = row['timestamp']

            if counter != 0 and row['timestamp'] - start_time > period:
                start_time = row['timestamp']
                print ''

                feature_set_classical = self.gen_feature_set(current_rows, classical=True, verbose=True)

                feature_set, c_vect = self.gen_feature_set(current_rows, classical=False, c_vect=c_vect, verbose=True)

                self.featurer.preserve()

                classical_admission = [True] * len(feature_set_classical)

                classical_features = [item.tolist() for item in feature_set_classical]

                predictions_classic_adm = {'GDSF': classical_admission,
                                           'LRU': classical_admission,
                                           'Oracle': classical_admission}

                predictions_classic_evc = {'GDSF': classical_features[:, 0],
                                           'LRU': classical_features[:, 1],
                                           'Oracle': classical_features[:, 3]}

                predictions_evc = {
                    'ML': classical_features[:, 0],
                    'DET ML': classical_features[:, 0]
                }

                admission_predictions = self.model_admission.predict(feature_set, verbose=1, batch_size=4 * 4096)

                predictions_adm = {
                    'ML': [item == 1
                           for item in vectorized(admission_predictions)],
                    'DET ML': [np.argmax(item) == 1 for item in admission_predictions]
                }

                for alg in algorithms_ml.keys():
                    algorithms_ml[alg].reset()
                for alg in algorithms_classic.keys():
                    algorithms_classic[alg].reset()

                GameEnvironment.test_algorithms(algorithms_ml,
                                                algorithms_classic,
                                                predictions_adm,
                                                predictions_evc,
                                                predictions_classic_adm,
                                                predictions_classic_evc,
                                                current_rows,
                                                o_file_generator + '_' + str(file_counter),
                                                verbose,
                                                base_iteration=counter - period)
                file_counter += 1
                current_rows = [row]
            else:
                current_rows.append(row)
                if verbose:
                    sys.stdout.write('\r' + 'Collector iteration {:7d}'.format(counter + 1))
                    sys.stdout.flush()
            counter += 1

    @staticmethod
    def select_elites(states, actions, rewards, percentile, return_indexing=False):

        if percentile is not None:
            elite_indicies = [i for i in range(len(rewards)) if rewards[i] >= percentile]
        else:
            elite_indicies = [rewards.index(max(rewards))]
        elite_states = np.concatenate([states[i] for i in elite_indicies], axis=0)
        elite_actions = np.concatenate([actions[i] for i in elite_indicies], axis=0)
        if not return_indexing:
            return elite_states, elite_actions
        else:
            return elite_states, elite_actions, elite_indicies

    def run_and_train_ml(self,
                         n_sessions,
                         iterations,
                         t_max,
                         period,
                         filenames,
                         skip,
                         n_threads=10):

        self.run_and_train(
            n_sessions,
            iterations,
            t_max,
            period,
            filenames,
            skip,
            n_threads,
            False,
            False
        )

    def run_and_train_ad(self,
                         n_sessions,
                         iterations,
                         t_max,
                         period,
                         filenames,
                         skip,
                         n_threads=10):

        self.run_and_train(
            n_sessions,
            iterations,
            t_max,
            period,
            filenames,
            skip,
            n_threads,
            True,
            False
        )

    def run_and_train(self,
                      n_sessions,
                      iterations,
                      t_max,
                      period,
                      filenames,
                      skip,
                      n_threads=10,
                      evc_classical=False,
                      adm_classical=False):

        assert self.ofname is not None
        q_percentile_admission = 95
        q_percentile_eviction = 95

        s_actions_evc = np.diag(np.ones((self.last_dim,)))
        s_actions_adm = np.diag(np.ones((2,)))

        epochs = 1

        warmup_period = 0#500000

        t_max += warmup_period

        batch_size = 4096

        ofl = open('M7_DATA', 'w')

        for iteration in range(iterations):

            evc_acc_history = []
            evc_distribution_history = []

            adm_acc_history = []
            adm_distribution_history = []

            counter = 0
            current_rows = []

            c_vect = None

            algorithm = MLSimulator(self.cache_size)

            det_algorithm = MLSimulator(self.cache_size)

            algorithms_ml = {'ML': algorithm,
                             'DET ML': det_algorithm}

            refresh_val = -1

            algorithm.refresh_period =5#max(0, refresh_val - 1 - (refresh_val * iteration) / iterations)
            det_algorithm.refresh_period = 5#max(0, refresh_val - 1 - (refresh_val * iteration) / iterations)

            algorithms_classic = {'GDSF': GDSimulator(self.cache_size),
                                  'LRU': LRUSimulator(self.cache_size),
                                  'Oracle': LRUSimulator(self.cache_size)}

            print 'New iteration', iteration

            self.featurer.full_reset()

            skip_required = warmup_period != 0

            base_iteration = 0

            iterative = True
            iterator_bool = True
            train_admission = not adm_classical#iteration % 2 == 1
            train_eviction = not evc_classical#iteration % 2 == 0

            step = period
            save_rows = period - step

            feature_set = None
            feature_set_classical = None

            predictions_evc_test = {}
            predictions_adm_test = {}

            predictions_classic_evc = {}
            predictions_classic_adm = {}

            for row in GameEnvironment.iterate_dataset_over_all(filenames, skip):
                if counter > t_max:
                    break

                counter += 1

                current_rows.append(row)

                if (skip_required and len(current_rows) != warmup_period) or \
                        (not skip_required and len(current_rows) != period):
                    continue

                print 'Using', len(current_rows), 'items'

                if feature_set is None:
                    feature_set, c_vect = self.gen_feature_set(current_rows, verbose=True, c_vect=c_vect)
                    feature_set_classical = self.gen_feature_set(current_rows, classical=True)
                else:
                    new_feature_set, c_vect = self.gen_feature_set(current_rows[period - step: period],
                                                                   verbose=True, c_vect=c_vect)
                    new_feature_set_classical = self.gen_feature_set(current_rows[period - step: period],
                                                                     classical=True)
                    feature_set = feature_set[step:]
                    feature_set_classical = feature_set_classical[step:]
                    feature_set = np.concatenate([feature_set, new_feature_set], axis=0)
                    feature_set_classical = np.concatenate([feature_set_classical, new_feature_set_classical], axis=0)

                self.featurer.preserve()

                print 'Logical time', self.featurer.logical_time

                if not evc_classical:
                    predictions_evc = self.model.predict(feature_set, batch_size=batch_size)
                else:
                    predictions_evc = feature_set_classical
                if not adm_classical:
                    predictions_adm = self.model_admission.predict(feature_set, batch_size=batch_size)
                else:
                    predictions_adm = feature_set_classical

                traffic_arrived = sum([item['size'] for item in current_rows])
                time_diff = current_rows[len(current_rows) - 1]['timestamp'] - current_rows[0]['timestamp']

                seconds = time_diff % 60
                time_diff /= 60
                minutes = time_diff % 60
                time_diff /= 60
                hours = time_diff % 60
                time_diff = '{:2d}:{:2d}:{:2d}'.format(hours, minutes, seconds)
                time_diff = time_diff.replace(' ', '0')
                print 'Size arrived {:^15s} Time passed'.format(fsize(traffic_arrived)), time_diff

                admission_decisions = [True] * len(feature_set_classical)

                if not evc_classical:
                    predictions_evc_test = {'ML': [compute_rating(item, predictions_evc.shape[1])
                                                   for item in vectorized(predictions_evc)],
                                            'DET ML': [np.argmax(item) for item in predictions_evc]}
                else:
                    predictions_evc_test = {'ML': feature_set_classical[:, 0],
                                            'DET ML': feature_set_classical[:, 0]}

                if not adm_classical:
                    predictions_adm_test = {'ML': vectorized(predictions_adm),
                                            'DET ML': [np.argmax(item) for item in predictions_adm]}
                else:
                    predictions_adm_test = {'ML': admission_decisions,
                                            'DET ML': admission_decisions}

                predictions_classic_evc = {'GDSF': feature_set_classical[:, 0],
                                           'LRU': feature_set_classical[:, 1],
                                           'Oracle': feature_set_classical[:, 3]}

                predictions_classic_adm = {'GDSF': admission_decisions,
                                           'LRU': admission_decisions,
                                           'Oracle': admission_decisions}

                if skip_required:
                    print 'Warming up', warmup_period

                    GameEnvironment.test_algorithms(algorithms_ml,
                                                    algorithms_classic,
                                                    predictions_adm_test,
                                                    predictions_evc_test,
                                                    predictions_classic_adm,
                                                    predictions_classic_evc,
                                                    current_rows,
                                                    verbose=True,
                                                    base_iteration=base_iteration)

                    for key in algorithms_classic.keys():
                        algorithms_classic[key].reset()

                    for key in algorithms_ml.keys():
                        algorithms_ml[key].reset()

                    current_rows = []
                    feature_set = None
                    feature_set_classical = None
                    base_iteration += warmup_period

                    skip_required = False
                    continue

                for key in predictions_evc_test.keys():
                    predictions_evc_test[key] = predictions_evc_test[key][:step]
                    predictions_adm_test[key] = predictions_adm_test[key][:step]

                for key in predictions_classic_evc.keys():
                    predictions_classic_evc[key] = predictions_classic_evc[key][:step]
                    predictions_classic_adm[key] = predictions_classic_adm[key][:step]

                amlc = {}
                for key in algorithms_ml.keys():
                    amlc[key] = copy_object(algorithms_ml[key])
                    amlc[key].reset()

                acc = {}
                for key in algorithms_classic.keys():
                    acc[key] = copy_object(algorithms_classic[key])
                    acc[key].reset()

                GameEnvironment.test_algorithms(amlc,
                                                acc,
                                                predictions_adm_test,
                                                predictions_evc_test,
                                                predictions_classic_adm,
                                                predictions_classic_evc,
                                                current_rows[:step],
                                                verbose=True,
                                                base_iteration=base_iteration)

                bool_array = [[train_eviction and not evc_classical, train_admission and not adm_classical]] * 1

                states_adm, actions_adm, rewards_adm = [], [], []
                states, actions, rewards = [], [], []

                repetitions = len(bool_array)

                drop = True

                for repetition in range(repetitions):
                    local_train_eviction, local_train_admission = bool_array[repetition]

                    if drop:
                        states_adm, actions_adm, rewards_adm = [], [], []
                        states, actions, rewards = [], [], []

                    sessions = []
                    admission_skipped = not local_train_admission
                    eviction_skipped = not local_train_eviction
                    for i in tqdm(range(0, n_sessions, n_threads)):
                        steps = min(n_threads, n_sessions - i)
                        threads = [None] * steps
                        results = [None] * steps
                        for thread_number in range(min(n_threads,  steps)):
                            threads[thread_number] = threaded(generate_session_continious)(
                                predictions_evc,
                                predictions_adm,
                                current_rows,
                                algorithm,
                                eviction_deterministic=not local_train_eviction,
                                collect_eviction=local_train_eviction,
                                eviction_defined=evc_classical,
                                admission_deterministic=not local_train_admission,
                                collect_admission=local_train_admission,
                                admission_defined=adm_classical)

                        for thread_number in range(min(n_threads, steps)):
                            results[thread_number] = threads[thread_number].result_queue.get()
                        if len(sessions) > 200:
                            mr = np.percentile([item[4] for item in sessions], 75)
                            results = [item for item in results if item[4] > mr]
                        if results:
                            sessions += results

                    for s, a, sa, aa, re, ra in sessions:
                        states.append(s)
                        states_adm.append(sa)
                        actions.append(a)
                        actions_adm.append(aa)
                        rewards.append(re)
                        rewards_adm.append(ra)

                    if local_train_admission and np.std(rewards_adm) > 1e-6:
                        if q_percentile_admission is not None:
                            percentile_value = np.percentile(rewards_adm, q_percentile_admission)
                        else:
                            percentile_value = None
                        elite_states, elite_actions, indexing = self.select_elites(states_adm,
                                                                                   actions_adm,
                                                                                   rewards_adm,
                                                                                   percentile_value,
                                                                                   return_indexing=True)

                        if len(elite_actions) <= 100000:
                            indicies = range(len(elite_actions))
                        else:
                            indicies = np.random.choice(range(len(elite_actions)), 100000, replace=False)
                        adm_unique_sampled = get_unique_dict(elite_actions[indicies], range(2))
                        adm_distribution_history.append(np.asarray(adm_unique_sampled))
                        adm_unique_sampled = np.sum(adm_distribution_history, axis=0) * 1. / \
                                             len(adm_distribution_history)
                        adm_unique_sampled = [int(adm_unique_sampled[i]) if i % 2 == 0 else adm_unique_sampled[i]
                                              for i in range(len(adm_unique_sampled))]
                        rstr_random = ' '.join(['{:4d} : {:7.3f}%'
                                                for _ in range(len(adm_unique_sampled) / 2)]).format(
                            *adm_unique_sampled)

                        print 'Admission', rstr_random
                        adm_distribution_history = []

                        v = self.model_admission.fit(feature_set[elite_states], s_actions_adm[elite_actions],
                                                     epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)
                        adm_acc_history.append(v.history['acc'][0])

                        print 'Admission accuracy :', 100 * np.mean(adm_acc_history)

                        self.model_admission.save_weights('models/adm_' + self.ofname)

                        predictions_adm = self.model_admission.predict(feature_set, batch_size=batch_size)
                    else:
                        admission_skipped = True

                    if local_train_eviction and np.std(rewards) > 1e-6:
                        if q_percentile_eviction is not None:
                            percentile_value = np.percentile(rewards, q_percentile_eviction)
                        else:
                            percentile_value = None
                        elite_states, elite_actions, indexing = self.select_elites(states,
                                                                                    actions,
                                                                                   rewards,
                                                                                   percentile_value,
                                                                                   return_indexing=True)
                        #if local_train_admission:
                         #   self.cm.trainable = False

                        if len(elite_actions) <= 100000:
                            indicies = range(len(elite_actions))
                        else:
                            indicies = np.random.choice(range(len(elite_actions)), 100000, replace=False)

                        evc_unique_sampled = get_unique_dict(elite_actions[indicies], range(predictions_evc.shape[1]))
                        evc_distribution_history.append(np.asarray(evc_unique_sampled))
                        evc_unique_sampled = np.sum(evc_distribution_history, axis=0) * 1. / \
                                             len(evc_distribution_history)
                        evc_unique_sampled = [int(evc_unique_sampled[i]) if i % 2 == 0 else evc_unique_sampled[i]
                                              for i in range(len(evc_unique_sampled))]
                        rstr_random = ' '.join(['{:4d} : {:7.3f}%'
                                                for _ in range(len(evc_unique_sampled) / 2)]).format(
                            *evc_unique_sampled)

                        print 'Eviction', rstr_random
                        evc_distribution_history = []

                        v = self.model.fit(feature_set[elite_states], s_actions_evc[elite_actions],
                                       epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)
                        evc_acc_history.append(v.history['acc'][0])

                        print 'Eviction accuracy :', 100 * np.mean(evc_acc_history)

                        #if local_train_admission:
                         #   self.cm.trainable = True

                        self.model.save_weights('models/evc_' + self.ofname)# + '_' + str(iteration))

                        predictions_evc = self.model.predict(feature_set, batch_size=batch_size)
                    else:
                        eviction_skipped = True
                    if admission_skipped and eviction_skipped:
                        break

                    mean_reward_admission = 0
                    mean_reward_eviction = 0

                    percentile_eviction = 0
                    percentile_admission = 0

                    if local_train_admission:
                        mean_reward_admission = 100 * np.mean(rewards_adm)
                        if q_percentile_admission is not None:
                            percentile_admission = np.percentile(rewards_adm, q_percentile_admission)
                        else:
                            percentile_admission = max(rewards_adm)
                        percentile_admission *= 100

                    if local_train_eviction:
                        mean_reward_eviction = 100 * np.mean(rewards)
                        if q_percentile_eviction is not None:
                            percentile_eviction = np.percentile(rewards, q_percentile_eviction)
                        else:
                            percentile_eviction = max(rewards)
                        percentile_eviction *= 100
                    print_str = 'Iteration {:^4.1f}% Repetition {:3d} ' \
                                'Admission mean reward {:10.4f}% threshold {:10.4f}% ' \
                                'Eviction mean reward {:10.4f}% threshold {:10.4f}%'
                    print print_str.format(
                        100 * float(counter - 1) / t_max,
                        repetition + 1,
                        mean_reward_admission,
                        percentile_admission,
                        mean_reward_eviction,
                        percentile_eviction)

                if not evc_classical:
                    predictions_evc_test = {'ML': [compute_rating(item, predictions_evc.shape[1])
                                                   for item in vectorized(predictions_evc[:step])],
                                            'DET ML': [np.argmax(item) for item in predictions_evc[:step]]}

                if not adm_classical:
                    predictions_adm_test = {'ML': vectorized(predictions_adm[:step]),
                                            'DET ML': [np.argmax(item) for item in predictions_adm[:step]]}

                for key in algorithms_ml.keys():
                    algorithms_ml[key].reset()
                for key in algorithms_classic.keys():
                    algorithms_classic[key].reset()

                GameEnvironment.test_algorithms(algorithms_ml,
                                                algorithms_classic,
                                                predictions_adm_test,
                                                predictions_evc_test,
                                                predictions_classic_adm,
                                                predictions_classic_evc,
                                                current_rows[:step],
                                                verbose=True,
                                                base_iteration=base_iteration)

                base_iteration += step

                del states
                del states_adm
                del rewards
                del rewards_adm
                del actions
                del actions_adm

                current_rows = current_rows[step:]
            '''
            ofl.write('IT' + str(iteration) + ' RESULT :\n')
            if train_eviction:
                evc_unique_sampled = np.sum(evc_distribution_history, axis=0) * 1. / \
                                     len(evc_distribution_history)
                evc_unique_sampled = [int(evc_unique_sampled[i]) if i % 2 == 0 else evc_unique_sampled[i]
                                      for i in range(len(evc_unique_sampled))]
                rstr_random = ' '.join(['{:4d} : {:7.3f}%'
                                        for _ in range(len(evc_unique_sampled) / 2)]).format(*evc_unique_sampled)
                ofl.write('\tEviction ' + rstr_random + '\n')
                ofl.write('\t' + 'Eviction accuracy : ' + str(100 * np.mean(evc_acc_history)) + '\n')
            if train_admission:
                adm_unique_sampled = np.sum(adm_distribution_history, axis=0) * 1. / \
                                     len(adm_distribution_history)
                adm_unique_sampled = [int(adm_unique_sampled[i]) if i % 2 == 0 else adm_unique_sampled[i]
                                      for i in range(len(adm_unique_sampled))]
                rstr_random = ' '.join(['{:4d} : {:7.3f}%'
                                        for _ in range(len(adm_unique_sampled) / 2)]).format(*adm_unique_sampled)
                ofl.write('\tEviction ' + rstr_random + '\n')
                ofl.write('\t' + 'Admission accuracy : ' + str(100 * np.mean(adm_acc_history)) + '\n')
            names = algorithms_ml.keys() + algorithms_classic.keys()
            values = [100 * metric(alg) for alg in algorithms_ml.values()] + \
                     [100 * metric(alg) for alg in algorithms_classic.values()]
            spaces = [fsize(alg.free_space()) for alg in algorithms_ml.values()] + \
                     [fsize(alg.free_space()) for alg in algorithms_classic.values()]
            print_string = ' | '.join(['{:^6s} HR {:5.2f}% FS {:^6s}' for _ in range(len(names))])
            print_string = 'Iteration {:10d} ' + print_string
            subst_vals = [t_max]
            for i in range(len(names)):
                subst_vals.append(names[i])
                subst_vals.append(values[i])
                subst_vals.append(spaces[i])
            ofl.write('\t' + print_string.format(*subst_vals) + '\n')
            ofl.flush()
            '''
        ofl.close()

    def iterate_dataset(self, t_max=None):
        if self.data is None:
            names = ['timestamp', 'id', 'size', 'frequency', 'lasp_app', 'log_time',
                     'exp_recency', 'exp_log', 'future']
            types = [int, int, int, int, int, int, float, float, float]
            hdlr = open(self.filenames[self.skip], 'r').readlines()
            lines_converted = [line.split(' ') for line in hdlr]
            lines_converted = [[types[i](line[i]) for i in range(len(types))] for line in lines_converted]
            self.data = [dict(zip(names, item)) for item in lines_converted]
        counter = 0
        for row in self.data:
            if t_max is not None and counter > t_max:
                break
            counter += 1
            yield row

    @staticmethod
    def iterate_dataset_over_all(filenames, skip):
        for fname in filenames[skip:]:
            names = ['timestamp', 'id', 'size', 'frequency', 'lasp_app', 'log_time',
                     'exp_recency', 'exp_log']#, 'future']
            types = [int, int, int, int, int, int, float, float]#, float]
            hdlr = open(fname, 'r')

            for line in hdlr:
                lines_converted = line.split(' ')
                lines_converted = [types[i](lines_converted[i]) for i in range(len(types))]
                yield dict(zip(names, lines_converted))

            hdlr.close()
