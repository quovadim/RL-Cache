import numpy as np
from tqdm import tqdm


def get_feature_percentiles(feature):
    percs = [np.percentile(feature, 0)]
    for q in range(1, 101, 3):
        p = np.percentile(feature, q)
        if p in percs:
            continue
        percs.append(p)
    return percs


def split_feature(feature):
    perc_steps = 1
    percs = [i * 100 / perc_steps for i in range(perc_steps + 1)]
    percentiles = [np.percentile(feature, item) for item in percs]
    percentiles[0] -= 1
    percentiles[len(percentiles) - 1] += 1
    percentiles = list(np.unique(percentiles))
    percentiles = sorted(percentiles)
    return [(percentiles[i-1], percentiles[i]) for i in range(1, len(percentiles))]


class PacketFeaturer:
    def __init__(self, load_name=None):
        self.logical_time = 0
        self.real_time = 0
        self.was_seen = []
        self.names = ['timestamp', 'id', 'size', 'frequency', 'lasp_app', 'exp_recency', 'log_time', 'exp_log']
        fake_request = dict(zip(self.names, [1] * len(self.names)))
        self.fnum = len(self.get_packet_features_pure(fake_request))
        self.statistics = []
        print 'Real features', self.fnum
        if load_name is not None:
            feature_set = np.asarray(np.load(load_name))
            feature_set = feature_set[1000000:, :self.fnum]
            self.feature_mappings = []
            for i in range(self.fnum):
                self.feature_mappings.append(split_feature(feature_set[:, i]))
                statistics_arrays = [[]] * len(self.feature_mappings[i])
                for item in tqdm(feature_set[:, i]):
                    feature_index = self.get_feature_vector(i, item, return_index=True)
                    statistics_arrays[feature_index].append(item)
                statistics_vector = [(np.mean(item), np.std(item)) for item in statistics_arrays]
                self.statistics += statistics_vector

            self.dim = len(self.get_packet_features(fake_request))
        else:
            self.dim = 0
        print 'Features dim', self.dim

        self.preserved_logical_time = 0
        self.preserved_real_time = 0
        self.preserved_was_seen = []

    def reset(self):
        self.logical_time = self.preserved_logical_time
        self.real_time = self.preserved_real_time
        self.was_seen = self.preserved_was_seen

    def full_reset(self):
        self.logical_time = 0
        self.real_time = 0
        self.was_seen = []
        self.preserve()

    def preserve(self):
        self.preserved_logical_time = self.logical_time
        self.preserved_real_time = self.real_time
        self.preserved_was_seen = self.was_seen

    def update_packet_state(self, packet):
        self.logical_time += 1
        self.real_time = packet['timestamp']

    def update_packet_info(self, packet):
        self.was_seen.append(packet['id'])
        if len(self.was_seen) > 650:
            self.was_seen = self.was_seen[1:]

    def observation_flag(self, packet):
        return 1 if packet['frequency'] != 1 else 0

    def get_packet_features_classical(self, packet):
        feature_vector = [float(packet['frequency']) / (float(packet['size'])),
                          packet['timestamp'], self.observation_flag(packet),
                          float(packet['frequency'])]
        return np.asarray(feature_vector)

    def get_packet_features_pure(self, packet):
        feature_vector = []

        feature_vector.append(float(packet['size']))

        feature_vector.append(-np.log(1e-10 + float(packet['frequency']) / (1 + self.logical_time)))

        feature_vector.append(np.log(1e-10 + float(packet['frequency']) / float(packet['size'])))

        feature_vector.append(float(self.real_time - packet['lasp_app']))
        feature_vector.append(float(self.logical_time - packet['log_time']))
        feature_vector.append(float(packet['exp_recency']))
        feature_vector.append(float(packet['exp_log']))
        #was_seen = -1.
        #if packet['id'] in self.was_seen:
        #    was_seen = 1.
        #feature_vector.append(was_seen)

        features = np.asarray(feature_vector)
        return features

    def get_feature_vector(self, feature_index, feature_value, return_index=False):
        tlen = len(self.feature_mappings[feature_index])
        addition = [0] * tlen
        counter = 0
        for low, up in self.feature_mappings[feature_index]:
            if low <= feature_value <= up:
                addition[counter] = feature_value
                break
            counter += 1
        if return_index:
            return counter
        else:
            return np.asarray(addition)

    def get_packet_features_from_pure(self, pure_features):
        result = []
        for i in range(self.fnum):
            result += self.get_feature_vector(i, pure_features[i]).tolist()
        result_features = np.asarray(result)
        return result_features

    def apply_statistics(self, result_features):
        result = [0] * len(result_features)
        for i in range(len(result_features)):
            if result_features[i] != 0:
                result[i] = min(10, max(-10, (result_features[i] - self.statistics[i][0]) / (self.statistics[i][1])))
        return np.asarray(result)

    def get_packet_features(self, packet):
        feature_vector = self.get_packet_features_pure(packet)
        return self.apply_statistics(self.get_packet_features_from_pure(feature_vector))
