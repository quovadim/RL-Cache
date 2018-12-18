from environment.environment_aux import *
from collections import deque

import numpy as np
from tqdm import tqdm

feature_names = ['log size',
                 'log frequency',
                 #'gdsf',
                 #'frequency',
                 #'size',
                 #'recency',
                 #'exponential recency',
                 'log gdsf',
                 'log bhr',
                 'log time recency',
                 'log request recency',
                 'log exp time recency',
                 'log exp request recency']


def collect_features(output_filename, t_max, filenames):
    feature_matrix = []
    counter = 0

    summary = np.zeros((len(feature_names),))

    featurer = PacketFeaturer(None)

    output_file = open(output_filename, 'w')

    try:
        for row in iterate_dataset(filenames):
            if counter != 0 and counter % 50000 == 0:
                summary += np.sum(feature_matrix, axis=0)
                for item in feature_matrix:
                    output_file.write(' '.join([str(element) for element in item]) + '\n')
                feature_matrix = []
            if counter >= t_max:
                break
            featurer.update_packet_state(row)
            data = featurer.get_pure_features(row).tolist()
            feature_matrix.append(np.asarray(data))
            if counter != 0 and counter % 5000 == 0:
                d = featurer.fnum
                str_formatted = ' '.join(['{:^8s}: {:^6.4f}' for _ in range(d)])
                str_formatted = '{:8d}  ' + str_formatted
                mean_list = summary / counter
                name_list = [item[:10] for item in feature_names]
                common = [counter]
                for i in range(len(name_list)):
                    common.append(name_list[i])
                    common.append(mean_list[i])
                str_formatted = str_formatted.format(*(common))
                sys.stdout.write('\r' + str_formatted)
                sys.stdout.flush()
            featurer.update_packet_info(row)
            counter += 1
        print ''
    except KeyboardInterrupt:
        pass
    for item in feature_matrix:
        output_file.write(' '.join([str(element) for element in item]) + '\n')
    output_file.close()


def print_mappings(mappings):
    mappings_flat = [item[0] for item in mappings] + [mappings[len(mappings) - 1][1]]
    pdata = ['A: {:5.3f}'] * len(mappings_flat[:min(17, len(mappings_flat))])
    pstr = ' | '.join(pdata)
    print pstr.format(*mappings_flat)


def print_statistics(statistics):
    stat_vector = []
    p_vector = []
    for mean, std in statistics[:min(10, len(statistics))]:
        stat_vector.append(mean)
        stat_vector.append(std)
        p_vector.append('M: {:5.3f} V: {:5.3f}')
    print ' | '.join(p_vector).format(*stat_vector)


class PacketFeaturer:
    def __init__(self, config, verbose=True):
        self.logical_time = 0
        self.real_time = 0
        self.memory_vector = None

        self.preserved_logical_time = 0
        self.preserved_real_time = 0
        self.preserved_memory_vector = 0

        self.verbose = verbose
        names = ['timestamp', 'id', 'size', 'frequency', 'lasp_app',
                 'exp_recency', 'log_time', 'exp_log', 'entropy', 'future']
        self.fake_request = dict(zip(names, [1] * len(names)))
        self.fnum = len(self.get_pure_features(self.fake_request))
        self.statistics = []
        self.bias = 0
        self.normalization_limit = 0
        if self.verbose:
            print 'Real features\033[1m', self.fnum, '\033[0m'

        self.forget_lambda = 0

        self.feature_mappings = []
        self.dim = 0

        self.warmup = 0
        self.split_step = 0

        if config is not None:
            self.apply_config(config)

        if self.verbose:
            print 'Features dim\033[1m', self.dim, '\033[0m'

        self.classical = self.dim == 0

        if self.verbose:
            print 'Operates in', 'classical' if self.classical else 'ML', 'mode'

        self.full_reset()

    def save_statistics(self, config):
        with open(config['filename'], 'w') as f:
            pickle.dump([self.feature_mappings,
                         self.statistics,
                         self.split_step,
                         self.warmup], f)

    def print_statistics(self):
        lindex = 0
        for i in range(self.fnum):
            if self.verbose:
                print 'Doing\033[1m', feature_names[i], '\033[0m'
                print_mappings(self.feature_mappings[i])
                print_statistics(self.statistics[lindex:lindex + len(self.feature_mappings[i])])
            lindex += len(self.feature_mappings[i])

    def collect_statistics(self, config):
        data_raw = open(config['statistics'], 'r').readlines()[config['warmup']:]
        feature_set = np.zeros(shape=(len(data_raw), self.fnum))

        for i in tqdm(range(feature_set.shape[0])):
            feature_set[i] = np.array([float(item) for item in data_raw[i].split(' ')[:self.fnum]])

        for i in range(self.fnum):
            if self.verbose:
                print 'Doing\033[1m', feature_names[i], '\033[0m'
            self.feature_mappings.append(split_feature(feature_set[:, i], config['split step']))
            statistics_arrays = []
            for _ in range(len(self.feature_mappings[i])):
                statistics_arrays.append(deque([]))
            for item in tqdm(feature_set[:, i]):
                _, feature_index = self.__get_feature_vector(i, item)
                statistics_arrays[feature_index].append(item)
            statistics_vector = [(np.mean(item), np.std(item)) for item in statistics_arrays]
            self.statistics += statistics_vector

    def load_statistics(self, config):
        with open(config['filename'], 'r') as f:
            data = pickle.load(f)
            self.feature_mappings = data[0]
            self.statistics = data[1]
            self.warmup = config['warmup']
            self.split_step = config['split step']
            assert self.split_step == data[2]
            assert self.warmup == data[3]

    def apply_config(self, config):
        self.forget_lambda = config['lambda']
        self.warmup = config['warmup']
        self.split_step = config['split step']
        loading_failed = False
        self.normalization_limit = config['normalization limit']
        self.bias = config['bias']
        if self.verbose:
            print 'Bias\033[1m', self.bias, '\033[0mNormalization\033[1m', self.normalization_limit, '\033[0m'
        if config['load']:
            if self.verbose:
                print 'Loading...'
            try:
                self.load_statistics(config)
                if self.verbose:
                    print '\033[1m\033[92mSUCCESS\033[0m'
            except:
                if self.verbose:
                    print '\033[1m\033[91mFAIL\033[0m'
                loading_failed = True

        if not config['load'] or loading_failed:
            self.collect_statistics(config)
            if config['save'] or loading_failed:
                self.save_statistics(config)
        if config['show stat'] and self.verbose:
            self.print_statistics()
        self.dim = len(self.get_ml_features(self.fake_request))
        self.memory_vector = np.zeros(self.dim)

    def full_reset(self):
        self.logical_time = 0
        self.real_time = 0
        if self.dim != 0:
            self.memory_vector = np.zeros(self.dim)
        else:
            self.memory_vector = None
        self.preserve()

    def reset(self):
        self.logical_time = self.preserved_logical_time
        self.memory_vector = self.preserved_memory_vector
        self.real_time = self.preserved_real_time

    def preserve(self):
        self.preserved_logical_time = self.logical_time
        self.preserved_memory_vector = self.memory_vector
        self.real_time = self.preserved_real_time

    def update_packet_state(self, packet):
        self.logical_time += 1
        self.real_time = packet['timestamp']

    def update_packet_info(self, packet):
        pass

    def get_pure_features(self, packet):
        feature_vector = []

        feature_vector.append(np.log(1 + float(packet['size'])))

        feature_vector.append(-np.log(1e-4 + float(packet['frequency']) / (1 + self.logical_time)))

        #feature_vector.append(float(packet['frequency']) / float(packet['size']))

        #feature_vector.append(float(packet['frequency']) / (1 + self.logical_time))

        #feature_vector.append(float(packet['size']))

        #feature_vector.append(self.real_time - packet['lasp_app'])

        #feature_vector.append(float(packet['exp_recency']))

        feature_vector.append(-1 * feature_vector[1] - feature_vector[0])

        feature_vector.append(-1 * feature_vector[1] + feature_vector[0])

        feature_vector.append(np.log(2 + float(self.real_time - packet['lasp_app'])))
        feature_vector.append(np.log(2 + float(self.logical_time - packet['log_time'])))
        feature_vector.append(np.log(2 + float(packet['exp_recency'])))
        feature_vector.append(np.log(2 + float(packet['exp_log'])))

        features = np.asarray(feature_vector)
        return features

    def gen_feature_set(self, rows, pure=False):
        self.reset()
        counter = 0

        if pure:
            feature_matrix = np.zeros((len(rows), self.fnum))
        else:
            if self.classical:
                feature_matrix = np.zeros((len(rows), 6))
            else:
                feature_matrix = np.zeros((len(rows), self.dim))

        iterator = rows
        if self.verbose:
            iterator = tqdm(rows)

        for row in iterator:
            self.update_packet_state(row)
            if pure:
                feature_matrix[counter] = self.get_pure_features(row)
            else:
                if self.classical:
                    feature_matrix[counter] = self.get_static_features(row)
                else:
                    feature_matrix[counter] = self.get_ml_features(row)

            self.update_packet_info(row)
            counter += 1

        if not self.classical and not pure:
            memory_features = np.zeros((len(rows), self.dim))
            for i in range(len(rows)):
                memory_features[i] = self.memory_vector
                self.memory_vector = self.memory_vector * self.forget_lambda + \
                                     feature_matrix[i] * (1 - self.forget_lambda)
            feature_matrix = np.concatenate([feature_matrix, memory_features], axis=1)

        return np.asarray(feature_matrix)

    def get_static_features(self, packet):
        feature_vector = [float(packet['frequency']) / (float(packet['size'])),
                          self.logical_time,
                          packet['frequency'] != 1,
                          float(packet['frequency']) * (float(packet['size']) / (1 + self.logical_time)),
                          float(packet['frequency']) / (1 + self.logical_time),
                          True]

        return np.asarray(feature_vector)

    def get_ml_features(self, packet):
        feature_vector = self.get_pure_features(packet)
        return self.__get_packet_features_from_pure(feature_vector)

    def get_features(self, packet):
        if self.classical:
            return self.get_static_features(packet)
        else:
            return self.get_ml_features(packet)

    def __get_feature_vector(self, feature_index, feature_value):
        tlen = len(self.feature_mappings[feature_index])
        addition = [0] * tlen
        counter = 0
        mlow = self.feature_mappings[feature_index][0][0]
        if feature_value < mlow:
            addition[counter] = feature_value
            return addition, counter

        mhigh = self.feature_mappings[feature_index][tlen - 1][1]
        if feature_value > mhigh:
            addition[tlen - 1] = feature_value
            counter = tlen - 1
            return addition, counter

        for low, up in self.feature_mappings[feature_index]:
            if low <= feature_value <= up:
                addition[counter] = feature_value
                return addition, counter
            counter += 1

        print feature_index, feature_value, self.classical
        print self.statistics
        assert False

    def __get_packet_features_from_pure(self, pure_features):
        result = []
        statistics = []
        for i in range(self.fnum):
            feature_vector, index = self.__get_feature_vector(i, pure_features[i])
            stat_vector = [0] * len(feature_vector)
            stat_vector[index] = 1
            statistics += stat_vector
            result += feature_vector
        return self.__apply_statistics(result, statistics)

    def __apply_statistics(self, result_features, statistics):
        result = [0] * len(result_features)
        for i in range(len(result_features)):
            if statistics[i] != 0:
                result[i] = self.bias + min(1, max(-1, (result_features[i] - self.statistics[i][0]) / (
                        1e-4 + self.normalization_limit * self.statistics[i][1])))
        return np.asarray(result)