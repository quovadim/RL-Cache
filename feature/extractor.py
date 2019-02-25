from collections import deque

import sys
import numpy as np
from tqdm import tqdm
import pickle
from mmap import mmap


def iterate_dataset(filenames):
    for fname in filenames:
        names = PacketFeaturer.core_feature_names
        types = PacketFeaturer.feature_types
        with open(fname, 'r+b') as hdlr:
            maped_file = mmap(hdlr.fileno(), 0)

            for line in iter(maped_file.readline, ""):
                line_converted = line.split(' ')
                line_converted = [types[i](line_converted[i]) for i in range(len(line_converted))]
                line_converted += [0] * len(names[len(line_converted):])
                yield dict(zip(names, line_converted))


def get_trace_length(filenames):
    start_fname = filenames[0]
    names = PacketFeaturer.core_feature_names
    types = PacketFeaturer.feature_types
    hdlr = open(start_fname, 'r')

    line = hdlr.readline()
    line_converted = line.split(' ')
    line_converted = [types[i](line_converted[i]) for i in range(len(line_converted))]
    line_converted += [0] * len(names[len(line_converted):])
    start_data = dict(zip(names, line_converted))
    hdlr.close()

    end_fname = filenames[len(filenames) - 1]
    hdlr = open(end_fname, 'r')

    line = hdlr.readlines()
    line = line[len(line) - 1]
    line_converted = line.split(' ')
    line_converted = [types[i](line_converted[i]) for i in range(len(line_converted))]
    line_converted += [0] * len(names[len(line_converted):])
    end_data = dict(zip(names, line_converted))
    hdlr.close()
    return end_data['timestamp'] - start_data['timestamp']


def split_feature(feature, perc_steps):
    percs = [i * 100 / perc_steps for i in range(perc_steps + 1)]
    percentiles = [np.percentile(feature, item) for item in percs]
    percentiles[0] -= 1
    percentiles[len(percentiles) - 1] += 1
    percentiles = list(np.unique(percentiles))
    percentiles = sorted(percentiles)
    return [(percentiles[i-1], percentiles[i]) for i in range(1, len(percentiles))]


def collect_features(output_filename, t_max, filenames):
    feature_matrix = []
    counter = 0

    featurer = PacketFeaturer(None)
    summary = np.zeros((len(featurer.names),))

    output_file = open(output_filename, 'w')

    def shorten_name(x):
        return ''.join([item[0] for item in x.split(' ')])

    try:
        for row in iterate_dataset(filenames):
            if (counter != 0 and counter % 50000 == 0) or counter == t_max:
                summary += np.sum(feature_matrix, axis=0)
                for item in feature_matrix:
                    output_file.write(' '.join([str(element) for element in item]) + '\n')
                feature_matrix = []
            featurer.update_packet_state(row)
            data = featurer.get_pure_features(row).tolist()
            feature_matrix.append(np.asarray(data))
            if (counter != 0 and counter % 5000 == 0) or counter == t_max:
                d = featurer.feature_num
                str_formatted = ' '.join(['{:s}: \033[1m{:^5.3f}\033[0m' for _ in range(d)])
                str_formatted = '\033[1m{:d}K\033[0m ' + str_formatted
                mean_list = summary / counter
                name_list = [shorten_name(item) for item in featurer.names]
                common = [counter / 1000]
                for i in range(len(name_list)):
                    common.append(name_list[i])
                    common.append(mean_list[i])
                str_formatted = str_formatted.format(*common)
                sys.stdout.write('\r' + str_formatted)
                sys.stdout.flush()
            featurer.update_packet_info(row)
            counter += 1
            if counter == t_max:
                break
        print ''
    except KeyboardInterrupt:
        pass
    for item in feature_matrix:
        output_file.write(' '.join([str(element) for element in item]) + '\n')
    output_file.close()


def print_mappings(mappings):
    mappings_flat = [item[0] for item in mappings] + [mappings[len(mappings) - 1][1]]
    pdata = ['A: \033[1m{:5.3f}\033[0m'] * len(mappings_flat[:min(17, len(mappings_flat))])
    pstr = ' | '.join(pdata)
    print pstr.format(*mappings_flat)


def print_statistics(statistics):
    stat_vector = []
    p_vector = []
    for mean, std in statistics[:min(10, len(statistics))]:
        stat_vector.append(mean)
        stat_vector.append(std)
        p_vector.append('M: \033[1m{:5.3f}\033[0m V: \033[1m{:5.3f}\033[0m')
    print ' | '.join(p_vector).format(*stat_vector)


class PacketFeaturer:

    core_feature_names = ['timestamp', 'id', 'size', 'number of observations', 'last appearance',
                          'logical time', 'exponential recency', 'exponential logical time', 'entropy', 'future']

    feature_extractors = {
        'log size': lambda x, l, r: np.log(1 + float(x['size'])),
        'log frequency': lambda x, l, r: -np.log(1e-4 + float(x['number of observations']) / float(1 + l)),
        'log gdsf': lambda x, l, r: -np.log(1e-4 + float(x['number of observations'])) +
                                    np.log(1 + float(x['size'])),
        'log bhr': lambda x, l, r: np.log(1e-4 + float(x['number of observations'])) +
                                   np.log(1 + float(x['size'])),
        'log time recency': lambda x, l, r: np.log(2 + float(r - x['last appearance'])),
        'log request recency': lambda x, l, r: np.log(2 + float(l - x['logical time'])),
        'log exp time recency': lambda x, l, r: np.log(2 + float(x['exponential recency'])),
        'log exp request recency': lambda x, l, r: np.log(2 + float(x['exponential logical time'])),
        'entropy': lambda x, l, r: float(x['entropy']),
        'gdsf': lambda x, l, r: float(x['number of observations']) / float(x['size']),
        'frequency': lambda x, l, r: float(x['number of observations']) / float(1 + l),
        'number of observations': lambda x, l, r: float(x['number of observations']),
        'size': lambda x, l, r: float(x['size']),
        'time recency': lambda x, l, r: float(r) - float(x['last appearance']),
        'exp time recency': lambda x, l, r: float(x['exponential recency']),
        'request recency': lambda x, l, r: float(l - x['logical time']),
        'exp request recency': lambda x, l, r: float(x['exponential logical time'])
    }

    ml_feature_names = feature_extractors.keys()

    log_features = [key for key in feature_extractors.keys() if 'log ' in key]

    feature_types = [int, int, int, int, int, int, float, float, float, float, float]

    def __init__(self, config, verbose=True):
        self.logical_time = 0
        self.real_time = 0
        self.memory_vector = None

        self.preserved_logical_time = 0
        self.preserved_real_time = 0
        self.preserved_memory_vector = 0

        self.verbose = verbose

        self.names = PacketFeaturer.ml_feature_names

        self.dim = 0

        self.feature_num = len(self.names)

        self.pure_mode = False

        if verbose:
            print 'Packet featurer creation started'

        if config is not None:
            self.names = config["usable names"]
            self.feature_num = len(self.names)
            self.statistics = {}
            self.bias = 0
            self.normalization_limit = 0
            self.pure_mode = config['pure mode']
            if self.verbose:
                print 'Features', ' '.join(
                    ['{:d}: \033[1m"{:s}"\033[0m'.format(1 + i, self.names[i]) for i in range(len(self.names))])
                print 'Real features\033[1m', self.feature_num, '\033[0m'

            self.forget_lambda = 0

            self.feature_mappings = {}

            self.warmup = 0
            self.split_step = 0

            self.apply_config(config)

            if self.verbose:
                print 'Features dim\033[1m', self.dim, '\033[0m'

        self.classical = self.dim == 0

        if self.verbose:
            print 'Operates in\033[1m', 'classical' if self.classical else 'ML', '\033[0mmode'

        self.full_reset()

        if verbose:
            print 'Packet featurer created'

    def save_statistics(self, config):
        with open(config['filename'], 'w') as f:
            pickle.dump([self.feature_mappings,
                         self.statistics,
                         self.split_step,
                         self.warmup,
                         self.names], f)

    def collect_statistics(self, config):
        pv = self.verbose
        self.verbose = True
        data_raw = open(config['statistics'], 'r').readlines()[config['warmup']:]
        feature_set = np.zeros(shape=(len(data_raw), self.feature_num))

        if self.verbose:
            print 'Loading\033[1m data\033[0m'
        indexes_of_features = [PacketFeaturer.ml_feature_names.index(name) for name in self.names]
        for i in tqdm(range(feature_set.shape[0])):
            data_full = data_raw[i].split(' ')
            feature_set[i] = np.array([float(data_full[j]) for j in indexes_of_features])

        for i in range(len(self.names)):
            name = self.names[i]
            if self.verbose:
                print 'Doing\033[1m', name, '\033[0m'
            self.feature_mappings[name] = split_feature(feature_set[:, i], config['split step'])
            statistics_arrays = []
            for _ in range(len(self.feature_mappings[name])):
                statistics_arrays.append(deque([]))
            for item in tqdm(feature_set[:, i]):
                _, feature_index = self.__get_feature_vector(name, item)
                statistics_arrays[feature_index].append(item)
            statistics_vector = [(np.mean(item), np.std(item)) for item in statistics_arrays]
            self.statistics[name] = statistics_vector
            if config['show stat'] and self.verbose:
                print_mappings(self.feature_mappings[name])
                print_statistics(self.statistics[name])
        self.verbose = pv

    def load_statistics(self, config):
        with open(config['filename'], 'r') as f:
            data = pickle.load(f)
            self.feature_mappings = data[0]
            self.statistics = data[1]
            self.warmup = config['warmup']
            self.split_step = config['split step']
            assert self.names == data[4]
            assert len(self.feature_mappings) >= len(self.names)
            assert self.split_step == data[2]
            assert self.warmup == data[3]

    def apply_config(self, config):
        self.forget_lambda = config['lambda']
        self.warmup = config['warmup']
        self.logical_time = 0
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
            for name in self.names:
                self.dim += len(self.statistics[name])
        else:
            self.dim = len(self.names)

        if self.pure_mode:
            self.dim = len(self.names)

        minmax = {}
        for name in self.names:
            minmax[name] = []
            for a, b in self.feature_mappings[name]:
                minmax[name].append(a)
                minmax[name].append(b)

        mv_vals = [(1 + min(minmax[name]), max(minmax[name]) - 1) for name in self.names]
        self.init_vals = mv_vals

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
        return np.asarray([PacketFeaturer.feature_extractors[key](packet, self.logical_time, self.real_time)
                           for key in self.names])

    def gen_feature_set(self, rows, pure=False):
        self.reset()
        counter = 0

        if pure:
            feature_matrix = np.zeros((len(rows), self.feature_num))
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
        feature_vector = [float(packet['number of observations']) / (float(packet['size'])),
                          self.logical_time,
                          1 if packet['number of observations'] > 1.5 else 0,
                          (float(packet['future']) - 1) / (float(packet['size'])),
                          float(packet['number of observations']),
                          1]

        return np.asarray(feature_vector)

    def get_ml_features(self, packet):
        feature_vector = self.get_pure_features(packet)
        if self.pure_mode:
            return feature_vector
        return self.__get_packet_features_from_pure(feature_vector)

    def get_features(self, packet):
        if self.classical:
            return self.get_static_features(packet)
        else:
            return self.get_ml_features(packet)

    def __get_feature_vector(self, feature_name, feature_value):
        tlen = len(self.feature_mappings[feature_name])
        addition = [0] * tlen
        counter = 0
        mlow = self.feature_mappings[feature_name][0][0]
        if feature_value < mlow:
            addition[counter] = feature_value
            return addition, counter

        mhigh = self.feature_mappings[feature_name][tlen - 1][1]
        if feature_value > mhigh:
            addition[tlen - 1] = feature_value
            counter = tlen - 1
            return addition, counter

        for low, up in self.feature_mappings[feature_name]:
            if low <= feature_value <= up:
                addition[counter] = feature_value
                return addition, counter
            counter += 1

        print feature_name, feature_value, self.classical
        print self.statistics
        assert False

    def __get_packet_features_from_pure(self, pure_features):
        result = []
        statistics = {}
        for name in self.names:
            feature_index = self.names.index(name)
            feature_vector, index = self.__get_feature_vector(name, pure_features[feature_index])
            stat_vector = [0] * len(feature_vector)
            stat_vector[index] = 1
            statistics[name] = stat_vector
            result += feature_vector
        return self.__apply_statistics(result, statistics)

    def __apply_statistics(self, result_features, statistics):
        result = [0] * self.dim
        index = 0
        for name in self.names:
            for j in range(len(self.statistics[name])):
                if statistics[name][j] != 0:
                    result[index] = self.bias + min(1, max(-1, (result_features[index] - self.statistics[name][j][0]) / (
                        1e-4 + self.normalization_limit * self.statistics[name][j][1])))
                index += 1
        return np.asarray(result)
