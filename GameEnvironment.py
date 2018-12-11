from hurry.filesize import size as fsize
from FeatureExtractor import PacketFeaturer
from config_sanity import check_statistics_config, check_model_config

from environment_aux import *
from model import *

import tensorflow as tf

from tqdm import tqdm


class GameEnvironment:
    def __init__(self, config):
        self.config = config

        self.featurer = PacketFeaturer(load_json(self.config['training']['statistics']))

        self.wing_size = self.config['model']["wing size"]
        self.last_dim = self.wing_size * 2 + 1

        self.common_model = create_common_model(self.config['model'], self.featurer.dim)
        self.model_admission = create_admission_model(self.config['model'], self.featurer.dim, self.common_model)
        self.model_eviction = create_eviction_model(self.config['model'], self.featurer.dim, self.common_model)

    def test(self, o_file_generator):
        counter = 0
        current_rows = []
        file_counter = 0

        config = self.config['testing']

        filenames = collect_filenames(config['data folder'])

        featurer = PacketFeaturer(self.config['statistics'])

        cache_size = config['cache size']

        classes_names = config['algorithms']

        algorithms_data = {}
        algorithms = {}

        for class_name in classes_names:
            class_type, rng_adm, _, rng_evc, _, _ = name_to_class(class_name)
            algorithms_data[class_name] = name_to_class(class_name)
            algorithms[class_name] = class_type(cache_size)

        start_time = None

        for row in iterate_dataset(filenames):
            if counter > config['requests max']:
                break

            current_rows.append(row)

            if start_time is None:
                start_time = row['timestamp']

            if counter != 0 and row['timestamp'] - start_time > config['period']:
                start_time = row['timestamp']
                print ''

                classical_features = featurer.gen_feature_set(current_rows, classical=True)

                ml_features = featurer.gen_feature_set(current_rows, classical=False)

                featurer.preserve()

                decisions_adm, decisions_evc = generate_data_for_models(
                    algorithms.keys(),
                    algorithms_data,
                    classical_features,
                    ml_features,
                    self.model_admission,
                    self.model_eviction,
                    config['batch size']
                )

                if config['reset']:
                    for alg in algorithms.keys():
                        algorithms[alg].reset()

                test_algorithms(algorithms, decisions_adm, decisions_evc, current_rows,
                                config['alpha'],
                                output_file=o_file_generator + '_' + str(file_counter),
                                base_iteration=counter)
                file_counter += 1

                current_rows = []

            if counter % 100 == 0:
                sys.stdout.write('\r' + 'Collector iteration {:7d}'.format(counter + 1))
                sys.stdout.flush()
            counter += 1


def train(config, admission_path, eviction_path, load_admission, load_eviction, n_threads=10):
    config_model = check_model_config(config['model'], verbose=False)
    if config_model is None:
        return
    config_statistics = check_statistics_config(config_model['statistics'], verbose=False)
    if config_statistics is None:
        return

    np.random.seed(config['seed'])
    random.seed(config['seed'])
    tf.set_random_seed(config['seed'])

    featurer = PacketFeaturer(config_statistics)

    model_admission, model_eviction, common_model, last_dim = create_models(config_model, featurer.dim)

    if load_eviction:
        print 'Loading pretrained from', load_eviction
        model_eviction.load_weights(eviction_path)
    if load_admission:
        print 'Loading pretrained from', load_admission
        model_admission.load_weights(admission_path)

    filenames = collect_filenames(config['data folder'])

    cache_size = config['cache size'] * 1024 * 1024

    percentile_admission = config['percentile admission']
    percentile_eviction = config['percentile eviction']

    epochs = config['epochs']
    warmup_period = config['warmup']
    batch_size = config['batch size']
    runs = config['runs']
    period = config['period']
    repetitions = config['repetitions']
    drop = config['drop']
    samples = config['samples']
    alpha = config['session configuration']['alpha']
    classes_names = config['algorithms']

    step = period - config['overlap']
    overlap = config['overlap']

    s_actions_evc = np.diag(np.ones((last_dim,)))
    s_actions_adm = np.diag(np.ones((2,)))

    refresh_value = config['refresh value']
    warming_to_required_state = False
    additional_warming = 0
    empty_loops = 0

    skip_required = warmup_period != 0
    base_iteration = 0

    if config['iterative']:
        runs *= 2

    for iteration in range(runs):
        current_rows = []

        classes_names.append(config['target'] + '-DET')
        classes_names.append(config['target'] + '-RNG')
        special_keys = [config['target'] + '-DET', config['target'] + '-RNG']
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
                    algorithm_rng = class_type(cache_size)
                    algorithms[class_name] = algorithm_rng
                else:
                    algorithm_det = class_type(cache_size)
                    algorithms[class_name] = algorithm_det
            else:
                algorithms[class_name] = class_type(cache_size)
            algorithms_data[class_name] = name_to_class(class_name)

        assert algorithm_rng is not None and algorithm_det is not None

        if config['refresh policy'] == 'static':
            pass
        if config['refresh policy'] == 'monotonic':
            refresh_value = max(0, refresh_value - 2 * (refresh_value * iteration) / runs)

        algorithm_rng.refresh_period = refresh_value
        algorithm_det.refresh_period = refresh_value

        if warming_to_required_state:
            featurer.full_reset()
            skip_required = warmup_period != 0
            base_iteration = 0

        if config['iterative']:
            assert not eviction_classical and not admission_classical
            eviction_turn = 1
            if config['start iteration'] == 'E':
                eviction_turn = 0
            train_admission = iteration % 2 != eviction_turn
            train_eviction = iteration % 2 == eviction_turn
        else:
            train_admission = not admission_classical
            train_eviction = not eviction_classical

        if config['iterative']:
            print 'New run', str(iteration / 2) + '-' + str(iteration % 2), \
                'Admission' if train_admission else 'Eviction'
        else:
            print 'New run', iteration

        train_admission = train_admission and config['IP:train admission']
        train_eviction = train_eviction and config['IP:train eviction']

        ml_features = None
        classical_features = None

        addition_history = []

        steps_before_skip = config['refresh period']

        for row in iterate_dataset(filenames):

            if steps_before_skip == 0:
                break

            current_rows.append(row)

            if (skip_required and len(current_rows) != warmup_period) or \
                    (not skip_required and not warming_to_required_state and len(current_rows) != period) or \
                    (not skip_required and warming_to_required_state and len(current_rows) != step):
                continue

            if ml_features is None:
                ml_features = featurer.gen_feature_set(current_rows, classical=False)
                classical_features = featurer.gen_feature_set(current_rows, classical=True)
            else:
                extended_ml_features = featurer.gen_feature_set(current_rows[overlap: period], classical=False)
                extended_classical_features = featurer.gen_feature_set(current_rows[period - step: period],
                                                                           classical=True)
                ml_features = ml_features[step:]
                classical_features = classical_features[step:]
                ml_features = np.concatenate([ml_features, extended_ml_features], axis=0)
                classical_features = np.concatenate([classical_features, extended_classical_features], axis=0)

            featurer.preserve()
            print 'Logical time', featurer.logical_time

            decisions_adm, decisions_evc = generate_data_for_models(
                algorithms.keys(),
                algorithms_data,
                classical_features,
                ml_features,
                model_admission,
                model_eviction,
                batch_size
            )

            traffic_arrived = sum([item['size'] for item in current_rows])
            time_diff = current_rows[len(current_rows) - 1]['timestamp'] - current_rows[0]['timestamp']
            time_diff = to_ts(time_diff)

            print 'Size arrived {:^15s} Time passed'.format(fsize(traffic_arrived)), time_diff

            if skip_required or warming_to_required_state:
                if skip_required:
                    print 'Warming up', warmup_period
                else:
                    print 'Skipping', step, 'left', (additional_warming - empty_loops) * step

                test_algorithms(algorithms, decisions_adm, decisions_evc, current_rows, alpha,
                                base_iteration=base_iteration, special_keys=special_keys)

                if warming_to_required_state:
                    if empty_loops == additional_warming:
                        warming_to_required_state = False
                    else:
                        empty_loops += 1

                ml_features = None
                classical_features = None
                base_iteration += len(current_rows)

                current_rows = []

                if skip_required:
                    skip_required = False

                for key in algorithms.keys():
                    algorithms[key].reset()
                continue

            print 'Step', 1 + config['refresh period'] - steps_before_skip, \
                'out of', config['refresh period']

            steps_before_skip -= 1

            for key in algorithms.keys():
                algorithms[key].reset()

            algorithms_copy = {}
            algorithms_copy_states = {}
            for key in algorithms.keys():
                algorithms_copy[key] = copy_object(algorithms[key])
                algorithms_copy_states[key] = algorithms_copy[key].get_ratings()

            test_algorithms(algorithms_copy, decisions_adm, decisions_evc, current_rows[:step], alpha,
                            base_iteration=base_iteration, special_keys=special_keys)

            for key in algorithms_copy_states.keys():
                assert algorithms_copy_states[key] == algorithms[key].get_ratings()

            bool_array = [[train_eviction, train_admission]] * repetitions

            states_adm, actions_adm, rewards_adm = [], [], []
            states_evc, actions_evc, rewards_evc = [], [], []
            indicies_to_remove = []
            addition_history = []

            for repetition in range(repetitions):
                local_train_eviction, local_train_admission = bool_array[repetition]

                if drop:
                    states_adm, actions_adm, rewards_adm = [], [], []
                    states_evc, actions_evc, rewards_evc = [], [], []
                    addition_history = []
                else:
                    indicies_to_remove = []
                    for i in range(len(addition_history)):
                        if (repetition - addition_history[i]) == config['store period']:
                            indicies_to_remove.append(i)
                    assert repetition == 0 or len(indicies_to_remove) == samples
                    for index in sorted(indicies_to_remove, reverse=True):
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
                if not admission_classical:
                    if local_train_admission or repetition == 0:
                        predictions_admission = model_admission.predict(ml_features,
                                                                        batch_size=batch_size,
                                                                        verbose=0)
                else:
                    predictions_admission = decisions_adm[config['target'] + '-RNG']
                if not eviction_classical:
                    if local_train_eviction or repetition == 0:
                        predictions_eviction = model_eviction.predict(ml_features,
                                                                      batch_size=batch_size,
                                                                      verbose=0)
                else:
                    predictions_eviction = decisions_evc[config['target'] + '-RNG']

                print 'Repetition', repetition + 1, 'out of', repetitions

                samples_to_generate = samples
                if repetition == 0 and not drop:
                    samples_to_generate = samples * config['store period']
                for i in tqdm(range(0, samples_to_generate, n_threads)):
                    steps = min(n_threads, samples_to_generate - i)
                    threads = [None] * steps
                    results = [None] * steps
                    for thread_number in range(min(n_threads,  steps)):
                        threads[thread_number] = threaded(generate_session_continious)(
                            predictions_eviction,
                            predictions_admission,
                            current_rows,
                            algorithm_rng,
                            config['session configuration'],
                            step,
                            eviction_deterministic=eviction_classical,
                            collect_eviction=local_train_eviction,
                            eviction_defined=eviction_classical,
                            admission_deterministic=admission_classical,
                            collect_admission=local_train_admission,
                            admission_defined=admission_classical)

                    for thread_number in range(min(n_threads, steps)):
                        results[thread_number] = threads[thread_number].result_queue.get()
                    if config['dump sessions']:
                        if len(sessions) > config['dump limit']:
                            dump_percentile = np.percentile([item[4] for item in sessions], config['dump percentile'])
                            results = [item for item in results if item[4] > dump_percentile]
                    if results:
                        sessions += results
                        if repetition == 0 and not drop:
                            addition_history += [1 + i // samples - config['store period']] * len(results)
                        else:
                            addition_history += [repetition] * len(results)

                for se, ae, sa, aa, re, ra in sessions:
                    if local_train_admission:
                        states_adm.append(sa)
                        actions_adm.append(aa)
                        rewards_adm.append(ra)
                    if local_train_eviction:
                        states_evc.append(se)
                        actions_evc.append(ae)
                        rewards_evc.append(re)

                print 'Admission samples', len(rewards_adm), 'Eviction samples', len(rewards_evc)

                if local_train_admission:
                    train_model(percentile_admission, model_admission, rewards_adm, states_adm, actions_adm,
                                predictions_admission, ml_features, s_actions_adm, epochs, batch_size,
                                config['max samples'], 'Admission')

                    model_admission.save_weights(admission_path)

                if local_train_eviction:
                    train_model(percentile_eviction, model_eviction, rewards_evc, states_evc, actions_evc,
                                predictions_eviction, ml_features, s_actions_evc, epochs, batch_size,
                                config['max samples'], 'Eviction')

                    model_eviction.save_weights(eviction_path)

            decisions_adm, decisions_evc = generate_data_for_models(algorithms.keys(), algorithms_data,
                                                                    classical_features, ml_features, model_admission,
                                                                    model_eviction, batch_size)

            test_algorithms(algorithms, decisions_adm, decisions_evc, current_rows[:step], alpha,
                            base_iteration=base_iteration, special_keys=special_keys)

            del states_evc
            del states_adm
            del rewards_evc
            del rewards_adm
            del actions_evc
            del actions_adm

            current_rows = current_rows[step:]

            base_iteration += step

        if not config['iterative'] or iteration % 2 == 1:
            additional_warming += config['refresh period']
        warming_to_required_state = True
        empty_loops = 0
