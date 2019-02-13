source_experiment_folder = 'experiments/{:s}/{:s}'
source_dump_files = 'experiments/{:s}/{:s}_{:s}'
source_test_folder = 'tests/{:s}/{:s}'
source_data_folder = 'data/{:s}_rewarded'
source_graph_folder = 'graphs/{:s}/{:s}'
source_configuration_rules = 'configuration_info/{:s}'


def get_configuration_rules(rules):
    return source_configuration_rules.format(rules)


def get_train_name(experiment):
    return source_experiment_folder.format(str(experiment), 'train.json')


def get_test_name(experiment):
    return source_experiment_folder.format(str(experiment), 'test.json')


def get_model_name(experiment):
    return source_experiment_folder.format(str(experiment), 'model.json')


def get_statistics_name(experiment):
    return source_experiment_folder.format(str(experiment), 'statistics.json')


def get_intervals_name(experiment):
    return source_experiment_folder.format(str(experiment), 'intervals')


def get_test_dump_name(experiment, trace):
    return source_dump_files.format(str(experiment), 'dump', trace)


def get_admission_name(experiment):
    return source_experiment_folder.format(str(experiment), 'adm')


def get_eviction_name(experiment):
    return source_experiment_folder.format(str(experiment), 'evc')


def get_history_name(experiment):
    return source_experiment_folder.format(str(experiment), 'history.log')


def get_tests_name(experiment, test):
    return source_test_folder.format(str(experiment), str(test))


def get_graphs_name(experiment, test):
    return source_graph_folder.format(str(experiment), str(test))


def get_data_name(test):
    return source_data_folder.format(str(test))