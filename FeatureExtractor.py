import numpy as np
from tqdm import tqdm


def get_feature_percentiles(feature):
    percs = [np.percentile(feature, 0)]
    for q in range(1, 101, 5):
        p = np.percentile(feature, q)
        if p in percs:
            continue
        percs.append(p)
    return percs


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
            lnames = ['size', 'frequency', 'gdsf', 'recency', 'log_recency', 'exp_recency', 'exp_log', 'bin_log']
            for i in tqdm(range(self.fnum)):
                #self.feature_mappings.append(get_feature_percentiles(feature_set[:, i]))
                self.statistics.append([np.mean(feature_set[:, i]), np.std(feature_set[:, i])])
                #print lnames[i]
                #print get_feature_percentiles(feature_set[:, i])
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
                          float(packet['frequency']) * (float(packet['size'])) / (1 + self.logical_time)]
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
        was_seen = -1.
        if packet['id'] in self.was_seen:
            was_seen = 1.
        feature_vector.append(was_seen)

        features = np.asarray(feature_vector)
        return features

    def get_feature_vector(self, feature_index, feature_value):
        tlen = len(self.feature_mappings[feature_index]) + 2
        addition = [0] * tlen
        if feature_value <= self.feature_mappings[feature_index][0]:
            addition[0] = 1
            #return addition
        if feature_value >= self.feature_mappings[feature_index][len(self.feature_mappings[feature_index]) - 1]:
            addition[tlen - 1] = 1
            #return addition
        for j in range(1, len(self.feature_mappings[feature_index])):
            if self.feature_mappings[feature_index][j - 1] <= feature_value <= self.feature_mappings[feature_index][j]:
                addition[j] = 1
                #return addition
        return addition
        print self.feature_mappings[feature_index]
        print feature_value
        assert False

    def get_packet_features_from_pure(self, pure_features):
        return np.asarray([min(10, max(-10, (pure_features[i] - self.statistics[i][0]) / (self.statistics[i][1]))) for i in range(len(pure_features))])
        #return np.asarray(feature_vector)
        final_features = []
        for i in range(self.fnum):
            final_features += self.get_feature_vector(i, pure_features[i])
        return np.asarray(final_features)

    def get_packet_features(self, packet):
        feature_vector = self.get_packet_features_pure(packet)
        return self.get_packet_features_from_pure(feature_vector)
