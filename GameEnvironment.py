from os import listdir
from os.path import isfile, join
import sys
from hurry.filesize import size as fsize
from tqdm import tqdm
from FeatureExtractor import PacketFeaturer

from environment_aux import *
from model import *


class GameEnvironment:
    def __init__(self, config):
        self.config = config

        filepath = self.config['data folder']
        filenames = [(join(filepath, f), int(f.replace('.csv', ''))) for f in listdir(filepath)
                     if isfile(join(filepath, f)) and '.csv' in f and 'lock' not in f]
        filenames = sorted(filenames, key=lambda x: x[1])
        self.filenames = [item[0] for item in filenames]

        self.cache_size = self.config['cache size'] * 1024 * 1024

        self.featurer = PacketFeaturer(self.config['statistics'])

        self.wing_size = self.config['model']["wing size"]
        self.last_dim = self.wing_size * 2 + 1

        self.common_model = create_common_model(self.config['model'], self.featurer.dim)
        self.model_admission = create_admission_model(self.config['model'], self.featurer.dim, self.common_model)
        self.model_eviction = create_eviction_model(self.config['model'], self.featurer.dim, self.common_model)

    def test(self, o_file_generator):
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

        for row in iterate_dataset(self.filenames):
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

                test_algorithms(algorithms, decisions_adm, decisions_evc, current_rows,
                                self.config['testing']['alpha'],
                                output_file=o_file_generator + '_' + str(file_counter),
                                base_iteration=counter)
                file_counter += 1

                current_rows = []

            if counter % 100 == 0:
                sys.stdout.write('\r' + 'Collector iteration {:7d}'.format(counter + 1))
                sys.stdout.flush()
            counter += 1

    def train(self,
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
        alpha = self.config['training']['session configuration']['alpha']

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
            special_keys = [self.config['training']['target'] + '-DET', self.config['training']['target'] + '-RNG']
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
                refresh_value = max(0, refresh_value - 2 * (refresh_value * iteration) / runs)

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

            addition_history = []

            for row in iterate_dataset(self.filenames):
                if counter > requests_max:
                    break
                counter += 1

                current_rows.append(row)

                if (skip_required and len(current_rows) != warmup_period) or \
                        (not skip_required and len(current_rows) != period):
                    continue

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

                    test_algorithms(algorithms, decisions_adm, decisions_evc, current_rows, alpha,
                                    base_iteration=base_iteration, special_keys=special_keys)

                    for key in algorithms.keys():
                        algorithms[key].reset()

                    current_rows = []
                    ml_features = None
                    classical_features = None
                    base_iteration += warmup_period

                    skip_required = False
                    continue

                algorithms_copy = {}
                algorithms_copy_states = {}
                for key in algorithms.keys():
                    algorithms_copy[key] = copy_object(algorithms[key])
                    algorithms_copy_states[key] = algorithms_copy[key].get_ratings()
                    algorithms_copy[key].reset()

                test_algorithms(algorithms_copy, decisions_adm, decisions_evc, current_rows[:step], alpha,
                                base_iteration=base_iteration, special_keys=special_keys)

                for key in algorithms_copy_states.keys():
                    assert algorithms_copy_states[key] == algorithms[key].get_ratings()

                bool_array = [[train_eviction, train_admission]] * repetitions

                states_adm, actions_adm, rewards_adm = [], [], []
                states_evc, actions_evc, rewards_evc = [], [], []

                for repetition in range(repetitions):
                    local_train_eviction, local_train_admission = bool_array[repetition]

                    if drop:
                        states_adm, actions_adm, rewards_adm = [], [], []
                        states_evc, actions_evc, rewards_evc = [], [], []
                        addition_history = []
                    else:
                        indicies_to_remove = []
                        for i in range(len(addition_history)):
                            if repetition - addition_history[i] >= self.config['training']['store period']:
                                indicies_to_remove.append(i)
                        for index in indicies_to_remove:
                            if local_train_eviction:
                                del states_evc[index]
                                del actions_evc[index]
                                del rewards_evc[index]
                            if local_train_admission:
                                del states_adm[index]
                                del actions_adm[index]
                                del rewards_adm[index]
                            del addition_history[index]

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
                                self.config['training']['session configuration'],
                                eviction_deterministic=eviction_classical,
                                collect_eviction=local_train_eviction,
                                eviction_defined=eviction_classical,
                                admission_deterministic=admission_classical,
                                collect_admission=local_train_admission,
                                admission_defined=admission_classical)

                        for thread_number in range(min(n_threads, steps)):
                            results[thread_number] = threads[thread_number].result_queue.get()
                        if self.config['training']['dump sessions']:
                            if len(sessions) > self.config['training']['dump limit']:
                                dump_percentile = np.percentile([item[4] for item in sessions],
                                                                self.config['training']['dump percentile'])
                                results = [item for item in results if item[4] > dump_percentile]
                        if results:
                            sessions += results
                            addition_history += [repetition] * len(results)

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

                test_algorithms(algorithms, decisions_adm, decisions_evc, current_rows[:step], alpha,
                                base_iteration=base_iteration, special_keys=special_keys)

                base_iteration += step

                del states_evc
                del states_adm
                del rewards_evc
                del rewards_adm
                del actions_evc
                del actions_adm

                current_rows = current_rows[step:]
