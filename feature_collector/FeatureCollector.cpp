//
// Created by vadim on 6/04/18.
//

#include "FeatureCollector.h"

#include <algorithm>
#include <stdint.h>

using std::find;
using std::min;
using std::max;
using std::pair;

void FeatureCollector::update_packet_state(Packet* packet) {
    logical_time += 1;
    real_time = packet->timestamp;

    uint64_t period = 3600;
    if (real_time - last_update > period / 16) {
    	clear_data(period);
    	last_update = real_time;
    }
    fa = false;
    if (packets_observed.find(packet->id) == packets_observed.end()) {
    	packet_mapping.insert(pair<uint64_t, Packet*>(packet->id, packet));
        packets_observed.emplace(packet->id);
        total_appearances.insert(pair<uint64_t, uint64_t>(packet->id, 0));
        last_appearance.insert(pair<uint64_t, uint64_t>(packet->id, packet->timestamp));
        logical_time_appearance.insert(pair<uint64_t, uint64_t>(packet->id, logical_time));
        packet_sizes.insert(pair<uint64_t, uint64_t>(packet->id, packet->size));
        exp_recency.insert(pair<uint64_t, double>(packet->id, -1));
        exp_logical.insert(pair<uint64_t, double>(packet->id, -1));
    } else {
    	delete packet_mapping.at(packet->id);
    	packet_mapping.at(packet->id) = packet;
    	if (exp_logical.at(packet->id) < 0) {
    		exp_logical.at(packet->id) = logical_time - logical_time_appearance.at(packet->id);
    		exp_recency.at(packet->id) = packet->timestamp - last_appearance.at(packet->id);
    	} else {
    		exp_logical.at(packet->id) = exp_logical.at(packet->id) * 0.9 + double(logical_time - logical_time_appearance.at(packet->id)) * 0.1;
    		exp_recency.at(packet->id) = exp_recency.at(packet->id) * 0.9 + double(packet->timestamp - last_appearance.at(packet->id)) * 0.1;
    	}
    }
    total_appearances.at(packet->id) += 1;
}

void FeatureCollector::update_packet_info(Packet* packet) {

    last_appearance.at(packet->id) = packet->timestamp;
    logical_time_appearance.at(packet->id) = logical_time;
}

vector<int64_t> FeatureCollector::get_packet_features(uint64_t packet_id) {
    vector<int64_t> result;
    // Frequency
    int64_t freq = total_appearances.at(packet_id);
    result.push_back(freq);
    // Recency
    int64_t time_last_app = last_appearance.at(packet_id);
    result.push_back(time_last_app);
    // Logical recency
    int64_t log =  logical_time_appearance.at(packet_id);
    result.push_back(log);
    return result;
}

vector<double> FeatureCollector::get_packet_features_dbl(uint64_t packet_id) {
	vector<double> result;
	// Exp recency
	double exp_rec = exp_recency.at(packet_id);
	result.push_back(exp_rec);
	// Exp logical time
	double exp_log = exp_logical.at(packet_id);
	result.push_back(exp_log);
	return result;
}

void FeatureCollector::clear_data(uint64_t max_interval) {
	vector<uint64_t> to_delete;
	for(auto it = packets_observed.begin(); it != packets_observed.end(); it++) {
		if (real_time - last_appearance.at(*it) >= max_interval) {
			to_delete.push_back(*it);
		}
	}
	for(auto it = to_delete.begin(); it != to_delete.end(); it++) {
		packets_observed.erase(*it);
		packet_mapping.erase(*it);
		total_appearances.erase(*it);
		last_appearance.erase(*it);
		logical_time_appearance.erase(*it);
		packet_sizes.erase(*it);
		exp_recency.erase(*it);
		exp_logical.erase(*it);
	}
}

double FeatureCollector::get_gdsf_feature(uint64_t packet_id) {
    	double size_var = packet_sizes.at(packet_id);
    	double freq = double(total_appearances.at(packet_id));
    	return freq / size_var;
    }
