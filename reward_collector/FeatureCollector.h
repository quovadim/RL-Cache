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
#include <math.h>
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
using std::log2;

class FeatureCollector{
public:
	FeatureCollector() :
			logical_time(0)
	{}

    double get_packet_rating(Packet* packet);

private:
	deque<uint64_t> time_sequence;
	deque<uint64_t> id_sequence;
	map<uint64_t, uint64_t> latest_logical_time;
	map<uint64_t, uint64_t> meetings;
	map<uint64_t, double> ratings;

	uint64_t logical_time;
};
