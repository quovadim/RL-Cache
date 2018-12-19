//
// Created by vadim on 7/04/18.
//

#pragma once

#include <string>
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
