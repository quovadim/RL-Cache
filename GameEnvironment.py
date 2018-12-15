from FeatureExtractor import PacketFeaturer

from environment_aux import *
from model import *

import tensorflow as tf

from tqdm import tqdm
from hurry.filesize import size as fsize
import sys


def test(config, o_file_generator):
    current_rows = []
    file_counter = 0

    filenames = collect_filenames(config['data folder'])

    featurers = config['featurers']
    admission_models = config['admission']
    eviction_models = config['eviction']
    models = config['models']
    statistics = config['statistics']
    sizes = config['sizes']

    cache_size = config['cache size'] * 1024 * 1024

    classes_names = config['algorithms'].keys()

    algorithms = {}

    counter = 0

    special_keys = []

    for class_name in classes_names:
        class_type, rng_adm, _, rng_evc, _, _ = name_to_class(class_name)
        if rng_adm or rng_evc:
            special_keys.append(class_name)
        algorithms[class_name] = class_type(sizes[class_name])

    start_time = None

    warmup_length = config['warmup']
    needs_warmup = True

    for row in iterate_dataset(filenames):

        current_rows.append(row)

        if start_time is None:
            start_time = row['timestamp']

        if len(current_rows) % 1000 == 0:
            sys.stdout.write('\rCollected: ' + str(len(current_rows)))
            sys.stdout.flush()

        if (needs_warmup and len(current_rows) == warmup_length) \
                or (not needs_warmup and row['timestamp'] - start_time > config['period']):
            start_time = row['timestamp']

            feature_sets = []

            print '\rCollected: ' + str(len(current_rows))

            if config['reset']:
                for alg in algorithms.keys():
                    algorithms[alg].reset()

            for featurer in featurers:
                feature_set = featurer.gen_feature_set(current_rows)
                feature_sets.append(feature_set)

            for featurer in featurers:
                featurer.preserve()

            predictions_adm, decisions_adm, \
            predictions_evc, decisions_evc, _ = generate_data_for_models(feature_sets,
                                                                         statistics,
                                                                         admission_models,
                                                                         eviction_models,
                                                                         models,
                                                                         config['batch size'])

            log_file = o_file_generator + '_' + str(file_counter)
            if needs_warmup:
                log_file = None

            test_algorithms(algorithms, predictions_adm, decisions_adm, predictions_evc, decisions_evc, current_rows,
                            config['alpha'],
                            output_file=log_file,
                            base_iteration=counter,
                            special_keys=special_keys)

            if not needs_warmup:
                file_counter += 1
                counter += len(current_rows)

            current_rows = []

            needs_warmup = False


def train(config, admission_path, eviction_path, load_admission, load_eviction, n_threads=10, verbose=False, show=True):
    config_model = config['model']
    if config_model is None:
        return
    config_statistics = config['feature extractor']
    if config_statistics is None:
        return

    if 'mc' not in config_model.keys():
        config_model['mc'] = False

    monte_carlo = config_model['mc']

    np.random.seed(config['seed'])
    random.seed(config['seed'] + 1)
    tf.set_random_seed(config['seed'] + 2)

    featurer = PacketFeaturer(config_statistics, verbose=False)

    model_admission, model_eviction, common_model, last_dim = create_models(config_model, featurer.dim)

    if load_eviction:
        print 'Loading pretrained from', eviction_path
        model_eviction.load_weights(eviction_path)
    if load_admission:
        print 'Loading pretrained from', admission_path
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

    if 'duplications' not in config.keys():
        config['duplications'] = 0

    duplication = config['duplications']

    s_actions_evc = np.diag(np.ones((last_dim,)))
    s_actions_adm = np.diag(np.ones((2,)))

    refresh_value = config['refresh value']
    warming_to_required_state = False
    additional_warming = 0
    empty_loops = 0

    if config['iterative']:
        runs *= 2

    thread_seed = 2 * config['seed'] + 1

    if common_model is not None and verbose:
        common_model.summary()

    class_type, rng_adm, _, rng_evc, _, rng_type = name_to_class(config['target'])
    if rng_adm and verbose:
        model_admission.summary()
    if rng_evc and verbose:
        model_eviction.summary()

    iteration = 0
    initial_prediod = period / 8
    initial_runs = runs / 2
    initial_samples = samples / 2
    initial_cache_size = cache_size

    total_duplications = duplication

    print '\033[93m' + ''.join(['-'] * 180) + '\033[0m'

    while iteration <= runs:

        config['session configuration']['seed'] = random.randint(0, 1000000 - 1)

        print_string = 'Using cache with size \033[1m\033[94m{:^6s}-{:^6s}\033[0m ' \
                       'next_duplication in \033[1m{:^4d}-{:^4d}\033[0m ' \
                       'Duplications left \033[1m{:^4d}-{:^4d}\033[0m ' \
                       'period \033[1m{:^9d}-{:^9d}\033[0m ' \
                       'samples \033[1m{:^5d}-{:^5d}\033[0m ' \
                       'seed \033[1m{:6d}\033[0m'

        print print_string.format(fsize(cache_size), fsize(cache_size + initial_cache_size * duplication),
                                  runs - iteration, runs,
                                  duplication, total_duplications,
                                  period, period + initial_prediod * duplication,
                                  samples, samples + initial_samples * duplication,
                                  config['session configuration']['seed'])

        print '\033[93m' + ''.join(['-'] * 180) + '\033[0m'

        if duplication != 0 and iteration == runs:
            cache_size += initial_cache_size
            duplication -= 1
            iteration = 0
            runs += initial_runs
            period += initial_prediod
            overlap += initial_prediod
            samples += initial_samples
            refresh_value = config['refresh value']
            print '\033[91mDUPLICATION\033[0m'
            continue

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
            refresh_value = max(0, refresh_value - (refresh_value * iteration) / runs)

        algorithm_rng.refresh_period = refresh_value
        algorithm_det.refresh_period = refresh_value

        featurer.full_reset()
        skip_required = warmup_period != 0
        base_iteration = 0

        if config['iterative']:
            assert not eviction_classical and not admission_classical
            eviction_turn = None
            if config['start iteration'] == 'E':
                eviction_turn = 0
            if config['start iteration'] == 'A':
                eviction_turn = 1
            assert eviction_turn is not None
            train_admission = iteration % 2 != eviction_turn
            train_eviction = iteration % 2 == eviction_turn
            if common_model is not None:
                common_model.trainable = iteration % 2 == 0
                model_admission = compile_model(model_admission, config_model, 'A')
                model_eviction = compile_model(model_eviction, config_model, 'E')
        else:
            train_admission = not admission_classical
            train_eviction = not eviction_classical

        if verbose:
            if config['iterative']:
                print 'New run\033[1m', str(iteration / 2) + '-' + str(iteration % 2),\
                    'Admission' if train_admission else 'Eviction', '\033[0m'

        train_admission = train_admission and config['IP:train admission']
        train_eviction = train_eviction and config['IP:train eviction']

        ml_features = None
        classical_features = None

        steps_before_skip = config['refresh period']

        start_time = None
        end_time = 0
        total_size = 0

        for row in iterate_dataset(filenames):

            if steps_before_skip == 0:
                break

            current_rows.append(row)

            if (skip_required and len(current_rows) != warmup_period) or \
                    (not skip_required and not warming_to_required_state and len(current_rows) != period) or \
                    (not skip_required and warming_to_required_state and len(current_rows) !=
                     step * config['refresh period']):
                continue

            if ml_features is None:
                featurer.classical = False
                ml_features = featurer.gen_feature_set(current_rows)
                featurer.classical = True
                classical_features = featurer.gen_feature_set(current_rows)
            else:
                featurer.classical = False
                extended_ml_features = featurer.gen_feature_set(current_rows[overlap:])
                featurer.classical = True
                extended_classical_features = featurer.gen_feature_set(current_rows[overlap:])
                ml_features = np.concatenate([ml_features, extended_ml_features], axis=0)
                classical_features = np.concatenate([classical_features, extended_classical_features], axis=0)

            featurer.preserve()

            decisions_adm, decisions_evc = generate_data_for_models_light(
                algorithms.keys(),
                algorithms_data,
                classical_features,
                ml_features,
                model_admission,
                model_eviction,
                batch_size
            )

            if not skip_required:
                if start_time is None:
                    start_time = current_rows[0]['timestamp']
                if warming_to_required_state:
                    total_size += sum([item['size'] for item in current_rows])
                    end_time = current_rows[len(current_rows) - 1]['timestamp']
                else:
                    total_size += sum([item['size'] for item in current_rows[:step]])
                    end_time = current_rows[step]['timestamp']

            if skip_required or warming_to_required_state:
                verbose_print = True

                test_algorithms_light(algorithms, decisions_adm, decisions_evc,
                                      current_rows, alpha, None, special_keys,
                                      base_iteration=base_iteration, verbose=verbose_print)

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
                if not skip_required and not warming_to_required_state:
                    print '\033[93m' + ''.join(['-'] * 180) + '\033[0m'
                continue

            time_diff = to_ts(end_time - start_time)
            if verbose:
                print 'Size arrived \033[1m{:^15s}\033[0m Time passed\033[1m'.format(fsize(total_size)), \
                    time_diff, '\033[0m'

            if config['refresh period'] > 1 and verbose:
                print 'Step\033[1m', 1 + config['refresh period'] - steps_before_skip, \
                    '\033[0mout of\033[1m', config['refresh period'], '\033[0m'

            steps_before_skip -= 1

            for key in algorithms.keys():
                algorithms[key].reset()

            bool_array = [[train_eviction, train_admission]] * repetitions

            states_adm, actions_adm, rewards_adm = [], [], []
            states_evc, actions_evc, rewards_evc = [], [], []
            addition_history = []

            if verbose:
                algorithms_copy = {}
                for key in algorithms.keys():
                    algorithms_copy[key] = copy_object(algorithms[key])
                    algorithms_copy[key].reset()

                test_algorithms_light(algorithms_copy, decisions_adm, decisions_evc, current_rows, alpha,
                                      None, special_keys, base_iteration=base_iteration, print_at=step,
                                      verbose=verbose)

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

                if repetitions > 1 and (verbose or show):
                    print 'Repetition', repetition + 1, 'out of', repetitions

                a = 0
                e = 0

                samples_to_generate = samples
                if repetition == 0 and not drop:
                    samples_to_generate = samples
                iterator = range(0, samples_to_generate, n_threads)
                if show or samples_to_generate > n_threads * 20:
                    iterator = tqdm(iterator)
                for i in iterator:
                    steps = n_threads
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
                            thread_seed,
                            eviction_deterministic=eviction_classical,
                            collect_eviction=local_train_eviction,
                            eviction_defined=eviction_classical,
                            admission_deterministic=admission_classical,
                            collect_admission=local_train_admission,
                            admission_defined=admission_classical)

                        thread_seed += 1

                    for thread_number in range(min(n_threads, steps)):
                        results[thread_number] = threads[thread_number].result_queue.get()
                    if config['dump sessions']:
                        if len(sessions) > config['dump limit']:
                            dump_percentile = np.percentile([item[4] for item in sessions], config['dump percentile'])
                            results = [item for item in results if item[4] > dump_percentile]
                    if results:
                        sessions += results
                        if not drop:
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

                if verbose:
                    print 'Admission samples', len(rewards_adm), 'Eviction samples', len(rewards_evc)

                if local_train_admission:
                    a = train_model(percentile_admission, model_admission, rewards_adm, states_adm, actions_adm,
                                    predictions_admission, ml_features, s_actions_adm, epochs, batch_size,
                                    config['max samples'], 'Admission', monte_carlo, verbose)
                    model_admission.save_weights(admission_path)

                if local_train_eviction:
                    e = train_model(percentile_eviction, model_eviction, rewards_evc, states_evc, actions_evc,
                                    predictions_eviction, ml_features, s_actions_evc, epochs, batch_size,
                                    config['max samples'], 'Eviction', monte_carlo, verbose)
                    model_eviction.save_weights(eviction_path)

            decisions_adm, decisions_evc = generate_data_for_models_light(algorithms.keys(), algorithms_data,
                                                                          classical_features, ml_features,
                                                                          model_admission, model_eviction,
                                                                          batch_size)

            if verbose:
                algorithms_copy = {}
                for key in algorithms.keys():
                    algorithms_copy[key] = copy_object(algorithms[key])
                    algorithms_copy[key].reset()

                test_algorithms_light(algorithms_copy, decisions_adm, decisions_evc, current_rows, alpha,
                                      None, special_keys, base_iteration=base_iteration, print_at=step,
                                      verbose=verbose)

            for key in algorithms.keys():
                algorithms[key].reset()

            test_algorithms_light(algorithms, decisions_adm, decisions_evc, current_rows[:step], alpha,
                                  None, special_keys, base_iteration=base_iteration, verbose=False)

            if not verbose:
                if not show:
                    pstr = '\r'
                else:
                    pstr = ''
                data = []
                if train_admission:
                    pstr += '\033[92mAdmission\033[0m accuracy \033[1m{:5.2f}%\033[0m'
                    data.append(a)
                if train_eviction:
                    if train_admission:
                        pstr += ' '
                    pstr += '\033[92mEviction\033[0m accuracy \033[1m{:5.2f}%\033[0m'
                    data.append(e)
                pstr += ' Step \033[1m{:2d}\033[0m'
                data.append(config['refresh period'] - steps_before_skip)
                pstr += '\n' if show or steps_before_skip == 0 else ''
                sys.stdout.write(pstr.format(*data))
                sys.stdout.flush()

            del states_evc
            del states_adm
            del rewards_evc
            del rewards_adm
            del actions_evc
            del actions_adm

            current_rows = current_rows[step:]
            ml_features = ml_features[step:]
            classical_features = classical_features[step:]

            base_iteration += step

        for key in algorithms.keys():
            algorithms[key].reset()

        if not config['iterative'] or iteration % 2 == 1:
            additional_warming += 1
        warming_to_required_state = True
        empty_loops = 0
        iteration += 1

        print '\033[93m' + ''.join(['-'] * 180) + '\033[0m'
