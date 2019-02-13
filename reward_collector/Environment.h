//
// Created by vadim on 7/04/18.
//

#pragma once

#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <map>
#include <vector>
#include <stdint.h>
#include <deque>

#include "FeatureCollector.h"
#include "Packet.h"

using std::string;
using std::vector;
using std::map;
using std::deque;
using std::ostringstream;

class Environment {
public:
    Environment(FeatureCollector* _collector, bool _verbose) :
            verbose(_verbose),
            collector(_collector),
            latest_timestamp(0)
    {}

    ~Environment() {
        delete collector;
    }

    void calculate_features(string& prefix_input, string& prefix_output, uint64_t files);

    string v2k(uint64_t value, uint64_t divisor) {
	    if (!(value / divisor)) {
	        return " ";
	    }
	    value /= divisor;
	    if (!(value / divisor)) {
	        return "K";
	    }
	    value /= divisor;
	    if (!(value / divisor)) {
	        return "M";
	    }
	    value /= divisor;
	    if (!(value / divisor)) {
	        return "G";
	    }
	    value /= divisor;
	    if (!(value / divisor)) {
	        return "T";
	    }
	    value /= divisor;
	    if (!(value / divisor)) {
	        return "P";
	    }
	    value /= divisor;
	    if (!(value / divisor)) {
	        return "E";
	    }
	    return "Z";
	}

	double shrink(uint64_t value, uint64_t divisor) {
	    while (value) {
	        if (value / divisor < divisor)
	            return int(10 * double(value) / divisor) / 10.0;//int(double(10 * value * divisor) / divisor) / 10.0;
	        value /= divisor;
	    }
	    return 0;
	}

	string timestamp_cnv(uint64_t timestamp) {
	    uint64_t seconds = timestamp % 60;
	    timestamp /= 60;
	    uint64_t minutes = timestamp % 60;
	    timestamp /= 60;
	    uint64_t hours = timestamp % 60;
	    timestamp /= 24;
	    uint64_t days = timestamp;

	    ostringstream ss;
	    ss << days << "d:"
	    << std::setfill('0') << std::setw(2)
	    << hours << "h:" << std::setw(2)
	    << minutes << "m:" << std::setw(2)
	    << seconds << "s";

        return ss.str();

	}

    vector<double> history;
    vector<bool> admit_history;
    vector<uint64_t> size_history;
    vector<uint64_t> objects_history;

private:
    bool verbose;


    map<uint64_t, Packet*> packet_storage;

    vector<Packet*> iterate_csv(string filename);

    FeatureCollector* collector;
    uint64_t latest_timestamp;
};
