//
// Created by vadim on 7/04/18.
//

#include "Environment.h"
#include "Packet.h"
#include <sstream>
#include <string>
#include <istream>
#include <fstream>
#include <iostream>
#include <stdint.h>
#include <iomanip>
#include <math.h>

using std::string;
using std::ifstream;
using std::getline;
using std::istringstream;
using std::cout;
using std::cerr;
using std::endl;
using std::ios;
using std::pair;
using std::ofstream;
using std::to_string;
using std::log;
using std::setw;


double shrink(double value, int multiplier) {
    int integer = multiplier * value;
    return double(integer) / multiplier;
}


void Environment::calculate_features(string& prefix_input, string& prefix_output, uint64_t files) {
	uint64_t latest_time = 0;
	uint64_t tcounter = 0;

	for (uint64_t i = 0; i < files; i++) {
		vector<Packet*> dataset;
		dataset = iterate_csv(prefix_input + to_string(i) + ".csv");
		ofstream stream;
		stream.open(prefix_output + to_string(i) + ".csv");
		for (uint64_t j = 0; j < dataset.size(); j++) {
			Packet* current_packet = dataset[j];
			if (current_packet->timestamp < latest_time)
				continue;

			latest_time = current_packet->timestamp;
			collector->update_packet_state(current_packet);
			if (j % 1000 == 0) {
				cerr << "\rFile " << prefix_output + to_string(i) + ".csv" <<
				" Local " << std::setw(10) << j / 1000 << "K" <<
				" Total " << std::setw(10) << tcounter / 1000 << "K" <<
				" Story " << std::setw(10) << collector->total_items() / 1000 << "K";
			}

			stream << current_packet->timestamp << " " << current_packet->id << " " << current_packet->size;

			vector<int64_t> features = collector->get_packet_features(current_packet->id);
			for (uint64_t k = 0; k < features.size(); k++) {
				stream << " " << features[k];
			}
			vector<double> features_dbl = collector->get_packet_features_dbl(current_packet->id);
			for (uint64_t k = 0; k < features_dbl.size(); k++) {
				stream << " " << features_dbl[k];
			}
			stream << endl;
			collector->update_packet_info(current_packet);
			tcounter++;
		}
		stream.close();
	}
}

vector<Packet*> Environment::iterate_csv(string filename) {
    ifstream input(filename);
    vector<Packet*> result;
    uint64_t time, id, size;
    while(input >> time >> id >> size) {
        Packet* new_packet = new Packet(id, size, time);
        result.emplace_back(new_packet);
    }
    return result;
}
