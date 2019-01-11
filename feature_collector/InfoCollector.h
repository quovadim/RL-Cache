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
#include <string>
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
using std::string;
using std::log2;

class InfoCollector {
public:
	InfoCollector() :
	entropy_sum(0),
	requests(0),
	bytes(0),
	unique_requests(0),
	unique_bytes(0),
	start_date(0),
	end_date(0)
	{}

	void update_info(Packet* packet);

	uint64_t get_unique_bytes() {
	    return unique_bytes;
	}

	uint64_t get_unique_requests() {
	    return unique_requests;
	}

	uint64_t get_bytes() {
	    return bytes;
	}

	uint64_t get_requests() {
	    return requests;
	}

	double get_entropy() {
	    uint64_t N = requests;
	    return log2(N) - entropy_sum / N;
	}

private:

	map<uint64_t, uint64_t> frequency;

	double entropy_sum;

	uint64_t requests;
	uint64_t bytes;

	uint64_t unique_requests;
	uint64_t unique_bytes;

	uint64_t start_date;
	uint64_t end_date;
};
