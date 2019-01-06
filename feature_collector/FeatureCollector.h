//
// Created by vadim on 6/04/18.
//

#pragma once
#include <map>
#include <vector>
#include <stdint.h>
#include <iostream>
#include <unordered_set>
#include <deque>
#include "Packet.h"

using std::map;
using std::unordered_set;
using std::vector;
using std::max;
using std::pair;
using std::cerr;
using std::endl;
using std::min;
using std::deque;

class FeatureCollector{
public:
	FeatureCollector() :
			entropy(0),
			N(0),
			current_sum(0),
			collected(0),
			logical_time(0),
			real_time(0),
			last_update(0),
			fa(true)
	{}

	void update_packet_state(Packet* packet);
	void update_packet_info(Packet* packet);

	uint64_t total_items() {
		return packets_observed.size();
	}

	uint64_t last_dropped() {
		return last_update;
	}

	vector<int64_t> get_packet_features(uint64_t packet_id);
	vector<double> get_packet_features_dbl(uint64_t packet_id);

	double get_popularity(uint64_t packet_id) {
		try {
			return total_appearances.at(packet_id);
		} catch (const std::out_of_range& e) {
			return 0;
		}
	}

	uint64_t get_packet_size(uint64_t packet_id) {
		return packet_sizes.at(packet_id);
	}

	Packet* get_packet(uint64_t packet_id) {
		try {
			return packet_mapping.at(packet_id);
		} catch (const std::out_of_range& e) {
			cerr << endl << packet_id << endl;
			throw e;
		}
	}

	double get_gdsf_feature(uint64_t packet_id);

	void clear_data(uint64_t max_interval);

private:

	double entropy;

	int N;
	double current_sum;
	int collected;
	deque<uint64_t> observed;
	deque<uint64_t> entropy_time;
	map<uint64_t, int> frequencies;

	deque<uint64_t> time_sequence;
	deque<uint64_t> id_sequence;
	map<uint64_t, int> observations;

	unordered_set<uint64_t> packets_observed;

	map<uint64_t, uint64_t> last_appearance;
	map<uint64_t, uint64_t> total_appearances;
	map<uint64_t, uint64_t> logical_time_appearance;
	map<uint64_t, uint64_t> packet_sizes;

	map<uint64_t, double> exp_recency;
	map<uint64_t, double> exp_logical;

	map<uint64_t, Packet*> packet_mapping;

	uint64_t logical_time;
	uint64_t real_time;

	uint64_t last_update;

	bool fa;
};
