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
from FeatureExtractor import PacketFeaturer, iterate_dataset, gen_feature_set

from multiprocessing import Process, Queue
from copy import deepcopy

import pickle


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


alpha = 1


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
    if name_eviction == 'Oracle':
        class_type = LRUSimulator
        eviction_random = False
        eviction_index = 3
    if name_eviction == 'ML':
        class_type = MLSimulator
        eviction_random = True
        eviction_index = -1
    if name_eviction == 'ML GD':
        class_type = GDSimulator
        eviction_random = True
        eviction_index = -1
    if name_eviction == 'ML LRU':
        class_type = LRUSimulator
        eviction_random = True
        eviction_index = -1

    return class_type, admission_random, admission_index, eviction_random, eviction_index, randomness_type


def metric_funct(x, y):
    return x * alpha + y * (1 - alpha)


def metric(algorithm):
    return metric_funct(algorithm.byte_hit_rate(), algorithm.hit_rate())


def compute_rating(value, total):
    w = (total - 1) / 2
    return 2. ** (float(value - w))


def sampling(prob_matrix):
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
            eviction_decisions = sampling(predictions_evc)
        evc_decision_values = eviction_decisions
        eviction_decisions = [compute_rating(eviction_decisions[i], len(predictions_evc[i]))
                              for i in range(len(eviction_decisions))]
    else:
        eviction_decisions = predictions_evc

    if not admission_defined:
        if admission_deterministic:
            admission_decisions = np.argmax(predictions_adm, axis=1)
        else:
            admission_decisions = sampling(predictions_adm)
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
                decisions_adm[key] = classical_feature_set[:, adm_index]

        if rng_evc:
            if predictions_evc is None:
                predictions_evc = evc_model.predict(ml_feature_set, verbose=0, batch_size=batch_size)
                amax_evc = [compute_rating(np.argmax(item), predictions_evc.shape[1])
                            for item in predictions_evc]
            assert eng_type is not None
            if eng_type:
                decisions_evc[key] = [compute_rating(item, predictions_evc.shape[1])
                                      for item in sampling(predictions_adm)]
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
        percentile_value = np.percentile(rewards, percentile)
    else:
        percentile_value = None
    elite_states, elite_actions, indexing = select_elites(states,
                                                          actions,
                                                          rewards,
                                                          percentile_value,
                                                          return_indexing=True)

    data = sampling(predictions)
    unique_sampled = get_unique_dict(data, range(predictions.shape[1]))
    unique_sampled = [int(unique_sampled[i]) if i % 2 == 0 else unique_sampled[i]
                          for i in range(len(unique_sampled))]
    rstr_random = ' '.join(['{:4d} : {:7.3f}%'
                            for _ in range(len(unique_sampled) / 2)]).format(*unique_sampled)

    print label, rstr_random

    v = model.fit(features_embedding[elite_states], answers_embedding[elite_actions],
                  epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)

    print label, 'accuracy :', 100 * np.mean(v.history['acc'][0])


class GameEnvironment:
    def __init__(self, config):
        self.config = config
        self.cache_size = self.config['cache size']

        self.featurer = PacketFeaturer(load_name='auxiliary/features_primary.npy')

        self.wing_size = self.config['model']["wing size"]
        self.last_dim = self.wing_size * 2 + 1

        dropout_rate = self.config['model']['dropout rate']

        if self.config['model']['use common']:
            multiplier_common = self.config['model']['multiplier common']
            layers_common = self.config['model']['layers common']

        multiplier_each = self.config['model']['multiplier each']
        layers_each = self.config['model']['layers each']

        if self.config['model']['use common']:
            self.cm = common_model = Sequential()
            common_model.add(l.Dense(self.featurer.dim * multiplier_common, input_shape=(2 * self.featurer.dim,),
                                     activation='elu'))
            if self.config['model']['use batch normalization']:
                common_model.add(l.BatchNormalization())
            for _ in range(layers_common):
                common_model.add(l.Dropout(dropout_rate))
                common_model.add(l.Dense(self.featurer.dim * multiplier_common, activation='elu'))

        self.model_eviction = Sequential()
        self.model_eviction.add(l.Dense(self.featurer.dim * multiplier_each, input_shape=(2 * self.featurer.dim,),
                                        activation='elu'))
        if self.config['model']['use common']:
            self.model_eviction.add(common_model)
        else:
            if self.config['model']['use batch normalization']:
                self.model_eviction.add(l.BatchNormalization())

        for i in range(layers_each):
            self.model_eviction.add(l.Dropout(dropout_rate))
            self.model_eviction.add(l.Dense(self.featurer.dim * int(multiplier_each * (layers_each - i) / layers_each),
                                            activation='elu'))
        self.model_eviction.add(l.Dropout(dropout_rate))
        self.model_eviction.add(l.Dense(self.last_dim, activation='softmax'))

        self.evc_optimizer = Adam(lr=self.config['model']['eviction lr'])

        self.model_eviction.compile(self.evc_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        self.model_admission = Sequential()

        self.model_admission.add(l.Dense(self.featurer.dim * multiplier_each, input_shape=(2 * self.featurer.dim,),
                                         activation='elu'))
        if self.config['model']['use common']:
            self.model_admission.add(common_model)
        else:
            if self.config['model']['use batch normalization']:
                self.model_admission.add(l.BatchNormalization())

        for i in range(layers_each):
            self.model_admission.add(l.Dropout(dropout_rate))
            self.model_admission.add(l.Dense(self.featurer.dim * int(multiplier_each * (layers_each - i) / layers_each),
                                             activation='elu'))

        self.model_admission.add(l.Dropout(dropout_rate))
        self.model_admission.add(l.Dense(2, activation='softmax'))

        self.adm_optimizezr = Adam(lr=self.config['model']['admission lr'])

        self.model_admission.compile(self.adm_optimizezr, loss='binary_crossentropy', metrics=['accuracy'])

    @staticmethod
    def test_algorithms(algorithms,
                        predictions_adm,
                        predictions_evc,
                        rows,
                        output_file=None,
                        base_iteration=0):

        counter = 0

        total_size = sum([row['size'] for row in rows])
        total_time = rows[len(rows) - 1]['timestamp'] - rows[0]['timestamp']

        history = {'time': [], 'size': 0}
        for key in algorithms.keys():
            history[key] = []

        for row in rows:
            for alg in algorithms.keys():
                algorithms[alg].decide(row, predictions_evc[alg][counter], predictions_adm[alg][counter])

            if output_file is not None and (counter % 100 < 0 or counter == len(rows) - 1):
                history['flow'] = float(total_size) / (1e-4 + float(total_time))
                history['time'].append(row['timestamp'])
                for key in algorithms.keys():
                    history[key].append(metric(algorithms[key]))

            if counter % 100 == 0 or counter == len(rows) - 1:
                names = algorithms.keys()
                values = [100 * metric(alg) for alg in algorithms.values()]
                spaces = [fsize(alg.free_space()) for alg in algorithms.values()]
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

        print ''

        if output_file is not None:
            pickle.dump(history, open(output_file, 'w'))

    def test(self, filenames, o_file_generator):
        counter = 0
        current_rows = []
        file_counter = 0

        classes_names = self.config['testing']['algorithms']

        algorithms_data = {}
        algorithms = {}

        for class_name in classes_names:
            class_type, rng_adm, _, rng_evc, _, _ = name_to_class(class_name)
            algorithms_data[class_name] = name_to_class(class_name)
            algorithms[class_name] = class_type(self.cache_size)

        memory_vector = None

        start_time = None

        forget_lambda = self.config['lambda features']

        for row in iterate_dataset(filenames):
            if counter > self.config['testing']['requests max']:
                break

            current_rows.append(row)

            if start_time is None:
                start_time = row['timestamp']

            if counter != 0 and row['timestamp'] - start_time > self.config['testing']['period']:
                start_time = row['timestamp']
                print ''

                classical_features = gen_feature_set(current_rows, self.featurer, forget_lambda, classical=True)

                ml_features, memory_vector = gen_feature_set(current_rows, self.featurer, forget_lambda,
                                                             classical=False, memory_vector=memory_vector)

                self.featurer.preserve()

                decisions_adm, decisions_evc = generate_data_for_models(
                    algorithms.keys(),
                    algorithms_data,
                    classical_features,
                    ml_features,
                    self.model_admission,
                    self.model_eviction,
                    self.config['batch size']
                )

                if self.config['testing']['reset']:
                    for alg in algorithms.keys():
                        algorithms[alg].reset()

                GameEnvironment.test_algorithms(algorithms,
                                                decisions_adm,
                                                decisions_evc,
                                                current_rows,
                                                output_file=o_file_generator + '_' + str(file_counter),
                                                base_iteration=counter)
                file_counter += 1

                current_rows = []

            if counter % 100 == 0:
                sys.stdout.write('\r' + 'Collector iteration {:7d}'.format(counter + 1))
                sys.stdout.flush()
            counter += 1

    def train(self,
              filenames,
              output_suffix,
              n_threads=10):

        percentile_admission = self.config['training']['percentile admission']
        percentile_eviction = self.config['training']['percentile eviction']

        epochs = self.config['training']['epochs']
        warmup_period = self.config['training']['warmup']
        batch_size = self.config['batch size']
        runs = self.config['training']['runs']
        requests_max = self.config['training']['requests max']
        period = self.config['training']['period']
        forget_lambda = self.config['lambda features']
        repetitions = self.config['training']['repetitions']
        drop = self.config['training']['drop']
        samples = self.config['training']['samples']

        s_actions_evc = np.diag(np.ones((self.last_dim,)))
        s_actions_adm = np.diag(np.ones((2,)))

        requests_max += warmup_period

        refresh_value = self.config['training']['refresh value']

        for iteration in range(runs):
            counter = 0
            current_rows = []
            memory_vector = None

            classes_names = self.config['training']['algorithms']
            classes_names.append(self.config['training']['target'] + '-DET')
            classes_names.append(self.config['training']['target'] + '-RNG')
            algorithms = {}
            algorithms_data = {}

            algorithm_rng = None
            algorithm_det = None

            eviction_classical = True
            admission_classical = True

            for class_name in classes_names:
                class_type, rng_adm, _, rng_evc, _, rng_type = name_to_class(class_name)
                if rng_adm or rng_evc:
                    if rng_type:
                        eviction_classical = not rng_evc
                        admission_classical = not rng_adm
                        algorithm_rng = class_type(self.cache_size)
                        algorithms[class_name] = algorithm_rng
                    else:
                        algorithm_det = class_type(self.cache_size)
                        algorithms[class_name] = algorithm_det
                else:
                    algorithms[class_name] = class_type(self.cache_size)
                algorithms_data[class_name] = name_to_class(class_name)

            assert algorithm_rng is not None and algorithm_det is not None

            if self.config['training']['refresh policy'] == 'static':
                pass
            if self.config['training']['refresh policy'] == 'monotonic':
                refresh_value = max(0, refresh_value - 1 - (refresh_value * iteration) / runs)

            algorithm_rng.refresh_period = refresh_value
            algorithm_det.refresh_period = refresh_value

            print 'New iteration', iteration

            self.featurer.full_reset()

            skip_required = warmup_period != 0
            base_iteration = 0

            if self.config['training']['iterative']:
                assert not eviction_classical and not admission_classical
                eviction_turn = 1
                if self.config['training']['start iteration'] == 'E':
                    eviction_turn = 0
                train_admission = iteration % 2 == eviction_turn
                train_eviction = iteration % 2 != eviction_turn
            else:
                train_admission = not admission_classical
                train_eviction = not eviction_classical

            train_admission = train_admission and self.config['training']['IP:train admission']
            train_eviction = train_eviction and self.config['training']['IP:train eviction']

            step = period - self.config['training']['overlap']
            overlap = self.config['training']['overlap']

            ml_features = None
            classical_features = None

            for row in iterate_dataset(filenames):
                if counter > requests_max:
                    break
                counter += 1

                current_rows.append(row)

                if (skip_required and len(current_rows) != warmup_period) or \
                        (not skip_required and len(current_rows) != period):
                    continue

                print 'Using', len(current_rows), 'items'

                if ml_features is None:
                    ml_features, memory_vector = gen_feature_set(current_rows, self.featurer, forget_lambda,
                                                                 classical=False, memory_vector=memory_vector)
                    classical_features = gen_feature_set(current_rows, self.featurer, forget_lambda, classical=True)
                else:
                    extended_ml_features, memory_vector = gen_feature_set(current_rows[overlap: period],
                                                                          self.featurer, forget_lambda,
                                                                          memory_vector=memory_vector)
                    extended_classical_features = gen_feature_set(current_rows[period - step: period],
                                                                  self.featurer, forget_lambda,
                                                                  classical=True)
                    ml_features = ml_features[step:]
                    classical_features = classical_features[step:]
                    ml_features = np.concatenate([ml_features, extended_ml_features], axis=0)
                    classical_features = np.concatenate([classical_features, extended_classical_features], axis=0)

                self.featurer.preserve()

                print 'Logical time', self.featurer.logical_time

                decisions_adm, decisions_evc = generate_data_for_models(
                    algorithms.keys(),
                    algorithms_data,
                    classical_features,
                    ml_features,
                    self.model_admission,
                    self.model_eviction,
                    self.config['batch size']
                )

                traffic_arrived = sum([item['size'] for item in current_rows])
                time_diff = current_rows[len(current_rows) - 1]['timestamp'] - current_rows[0]['timestamp']
                time_diff = to_ts(time_diff)

                print 'Size arrived {:^15s} Time passed'.format(fsize(traffic_arrived)), time_diff

                if skip_required:
                    print 'Warming up', warmup_period

                    GameEnvironment.test_algorithms(algorithms,
                                                    decisions_adm,
                                                    decisions_evc,
                                                    current_rows,
                                                    base_iteration=base_iteration)

                    for key in algorithms.keys():
                        algorithms[key].reset()

                    current_rows = []
                    ml_features = None
                    classical_features = None
                    base_iteration += warmup_period

                    skip_required = False
                    continue

                algorithms_copy = {}
                for key in algorithms.keys():
                    algorithms_copy[key] = copy_object(algorithms[key])
                    algorithms_copy[key].reset()

                GameEnvironment.test_algorithms(algorithms_copy,
                                                decisions_adm,
                                                decisions_evc,
                                                current_rows[:step],
                                                base_iteration=base_iteration)

                bool_array = [[train_eviction, train_admission]] * repetitions

                states_adm, actions_adm, rewards_adm = [], [], []
                states_evc, actions_evc, rewards_evc = [], [], []

                for repetition in range(repetitions):
                    local_train_eviction, local_train_admission = bool_array[repetition]

                    if drop:
                        states_adm, actions_adm, rewards_adm = [], [], []
                        states_evc, actions_evc, rewards_evc = [], [], []

                    sessions = []
                    admission_skipped = not local_train_admission
                    eviction_skipped = not local_train_eviction
                    predictions_admission = decisions_adm[self.config['training']['target'] + '-RNG']
                    if local_train_admission:
                        predictions_admission = self.model_admission.predict(ml_features,
                                                                             batch_size=batch_size,
                                                                             verbose=0)
                    predictions_eviction = decisions_evc[self.config['training']['target'] + '-RNG']
                    if local_train_eviction:
                        predictions_eviction = self.model_eviction.predict(ml_features,
                                                                           batch_size=batch_size,
                                                                           verbose=0)
                    for i in tqdm(range(0, samples, n_threads)):
                        steps = min(n_threads, samples - i)
                        threads = [None] * steps
                        results = [None] * steps
                        for thread_number in range(min(n_threads,  steps)):
                            threads[thread_number] = threaded(generate_session_continious)(
                                predictions_eviction,
                                predictions_admission,
                                current_rows,
                                algorithm_rng,
                                eviction_deterministic=not local_train_eviction,
                                collect_eviction=local_train_eviction,
                                eviction_defined=eviction_classical,
                                admission_deterministic=not local_train_admission,
                                collect_admission=local_train_admission,
                                admission_defined=eviction_classical)

                        for thread_number in range(min(n_threads, steps)):
                            results[thread_number] = threads[thread_number].result_queue.get()
                        if self.config['training']['dump sessions']:
                            if len(sessions) > self.config['training']['dump limit']:
                                dump_percentile = np.percentile([item[4] for item in sessions],
                                                                self.config['training']['dump percentile'])
                                results = [item for item in results if item[4] > dump_percentile]
                        if results:
                            sessions += results

                    for se, ae, sa, aa, re, ra in sessions:
                        states_evc.append(se)
                        states_adm.append(sa)
                        actions_evc.append(ae)
                        actions_adm.append(aa)
                        rewards_evc.append(re)
                        rewards_adm.append(ra)

                    if local_train_admission and np.std(rewards_adm) > 1e-6:
                        train_model(percentile_admission,
                                    self.model_admission,
                                    rewards_adm,
                                    states_adm,
                                    actions_adm,
                                    predictions_admission,
                                    ml_features,
                                    s_actions_adm,
                                    epochs,
                                    batch_size,
                                    'Admission')

                        self.model_admission.save_weights('models/adm_' + output_suffix)
                    else:
                        admission_skipped = True

                    if local_train_eviction and np.std(rewards_evc) > 1e-6:
                        train_model(percentile_eviction,
                                    self.model_eviction,
                                    rewards_evc,
                                    states_evc,
                                    actions_evc,
                                    predictions_eviction,
                                    ml_features,
                                    s_actions_evc,
                                    epochs,
                                    batch_size,
                                    'Eviction')

                        self.model_eviction.save_weights('models/evc_' + output_suffix)
                    else:
                        eviction_skipped = True

                    if admission_skipped and eviction_skipped:
                        break

                decisions_adm, decisions_evc = generate_data_for_models(
                    algorithms.keys(),
                    algorithms_data,
                    classical_features,
                    ml_features,
                    self.model_admission,
                    self.model_eviction,
                    self.config['batch size']
                )

                for key in algorithms.keys():
                    algorithms[key].reset()

                GameEnvironment.test_algorithms(algorithms,
                                                decisions_adm,
                                                decisions_evc,
                                                current_rows[:step],
                                                base_iteration=base_iteration)

                base_iteration += step

                del states_evc
                del states_adm
                del rewards_evc
                del rewards_adm
                del actions_evc
                del actions_adm

                current_rows = current_rows[step:]
