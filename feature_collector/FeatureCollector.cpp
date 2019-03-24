//
// Created by vadim on 6/04/18.
//

#include "FeatureCollector.h"

#include <algorithm>
#include <stdint.h>
#include <math.h>

using std::find;
using std::min;
using std::max;
using std::pair;
using std::log2;

void FeatureCollector::update_packet_state(Packet* packet) {
	logical_time += 1;
	real_time = packet->timestamp;

	uint64_t period = 3600 * 12;

	time_sequence.push_back(real_time);
	id_sequence.push_back(packet->id);
	if (observations.find(packet->id) != observations.end()) {
		observations[packet->id]++;
	} else {
	observations.insert(pair<uint64_t, int>(packet->id, 1));
	}
	uint64_t time_old = time_sequence.front();
	uint64_t id_to_remove = id_sequence.front();

	while ((real_time - time_old > period) && (time_sequence.size() >= 100000)) {

		observations[id_to_remove]--;
		if (observations[id_to_remove] == 0) {
			total_appearances.erase(id_to_remove);
			last_appearance.erase(id_to_remove);
			logical_time_appearance.erase(id_to_remove);
			packet_sizes.erase(id_to_remove);
			exp_recency.erase(id_to_remove);
			exp_logical.erase(id_to_remove);
			observations.erase(id_to_remove);
			delete packet_mapping.at(id_to_remove);
			packet_mapping.erase(id_to_remove);
			packets_observed.erase(id_to_remove);
		}

		time_sequence.pop_front();
		id_sequence.pop_front();

		time_old = time_sequence.front();
		id_to_remove = id_sequence.front();
	}

	const uint64_t collection_interval = 600;

	observed.push_back(packet->id);
	entropy_time.push_back(packet->timestamp);

    uint64_t entropy_id = observed.front();
	uint64_t entropy_timestamp = entropy_time.front();

	while (real_time - entropy_timestamp > collection_interval) {
	    //cerr << "Here " << entropy_id << " " << entropy_time.size() << endl;
	    entropy_time.pop_front();
		observed.pop_front(); // remove oldest element

		int n = frequencies.at(entropy_id); // get its frequency

		if (n != 1) {
			current_sum -= n * log2(n);
			current_sum += (n - 1) * log2(n - 1); // update Cs -n*log(n) + (n-1)log(n-1) so -old + new
			frequencies[entropy_id]--;
		} else {
			frequencies.erase(entropy_id); // Remove elements with 0 frequency, no use from them
		}
		entropy_id = observed.front();
	    entropy_timestamp = entropy_time.front();
	}

	if (frequencies.find(packet->id) != frequencies.end()) {
		int n = frequencies[packet->id];
		current_sum -= n * log2(n);
		current_sum += (n + 1) * log2(n + 1); // update Cs -n*log(n) + (n+1)log(n+1) so -old + new
		frequencies[packet->id]++;
	} else {
		frequencies.insert(pair<uint64_t, int>(packet->id, 1));
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

	double entropy = get_entropy();

	result.push_back(entropy);

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

	}
}

double FeatureCollector::get_gdsf_feature(uint64_t packet_id) {
	double size_var = packet_sizes.at(packet_id);
	double freq = double(total_appearances.at(packet_id));
	return freq / size_var;
}
