//
// Created by vadim on 6/04/18.
//

#include "FeatureCollector.h"

#include <algorithm>
#include <stdint.h>
#include <math.h>
#include <iostream>

using std::find;
using std::min;
using std::max;
using std::pair;
using std::log2;
using std::pow;
using std::cout;
using std::cerr;
using std::endl;

double FeatureCollector::get_packet_rating(Packet* packet) {
	logical_time += 1;

	uint64_t period = 7200;

	time_sequence.push_back(packet->timestamp);
	id_sequence.push_back(packet->id);

	double rating = 0;

	if (meetings.find(packet->id) != meetings.end()) {
		meetings[packet->id]++;
		rating = ratings[packet->id];
		double gamma = 0.999999;
		gamma = pow(gamma, logical_time - latest_logical_time[packet->id]);
		rating = 1 + rating * gamma;
		ratings[packet->id] = rating;
		latest_logical_time[packet->id] = logical_time;
	} else {
	    meetings.insert(pair<uint64_t, uint64_t>(packet->id, 1));
	    latest_logical_time.insert(pair<uint64_t, uint64_t>(packet->id, logical_time));
	    ratings.insert(pair<uint64_t, double>(packet->id, 1.));
	    rating = 1.;
	}

	uint64_t time_old = time_sequence.front();
	uint64_t id_to_remove = id_sequence.front();

	while (time_old - packet->timestamp > period && time_sequence.size() >= 100000) {

		meetings[id_to_remove]--;
		if (meetings[id_to_remove] == 0) {
			latest_logical_time.erase(id_to_remove);
			ratings.erase(id_to_remove);
			meetings.erase(id_to_remove);
		}

		time_sequence.pop_front();
		id_sequence.pop_front();

		time_old = time_sequence.front();
		id_to_remove = id_sequence.front();
	}
	return rating;
}