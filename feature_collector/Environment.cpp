//
// Created by vadim on 7/04/18.
//

#include "Environment.h"
#include "Packet.h"
#include "InfoCollector.h"
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


void Environment::calculate_features(string& prefix_input, string& prefix_output, uint64_t files) {
	uint64_t latest_time = 0;
	uint64_t tcounter = 0;

	InfoCollector info_collector;

	ofstream ch_stream;
	ch_stream.open(prefix_input + "../characteristics_" + to_string(files));

	uint64_t start_time = 0;

	for (uint64_t i = 0; i < files; i++) {
		vector<Packet*> dataset;
		dataset = iterate_csv(prefix_input + to_string(i) + ".csv");
		ofstream stream;
		stream.open(prefix_output + to_string(i) + ".csv");
		for (uint64_t j = 0; j < dataset.size(); j++) {
			Packet* current_packet = dataset[j];
			if (current_packet->timestamp < latest_time)
				continue;

			if (!start_time) {
			    start_time = current_packet->timestamp;
			}

			latest_time = current_packet->timestamp;

			collector->update_packet_state(current_packet);
			info_collector.update_info(current_packet);
			if (j % 1687 == 0) {
				cerr << std::fixed << std::setprecision(1) <<
				"\rFile " << to_string(i) + ".csv" <<
				" |  Time  " << std::setw(10) << timestamp_cnv(latest_time - start_time) <<
				" |  Local " << std::setw(6) << shrink(j, 1000) << v2k(j, 1000) <<
				" |  UB " << std::setw(10) << shrink(info_collector.get_unique_bytes(), 1024) <<
				v2k(info_collector.get_unique_bytes(), 1024) <<
				" |  B " << std::setw(10) << shrink(info_collector.get_bytes(), 1024) <<
				v2k(info_collector.get_bytes(), 1024) <<
				" |  UR " << std::setw(10) << shrink(info_collector.get_unique_requests(), 1000) <<
				v2k(info_collector.get_unique_requests(), 1000) <<
				" |  R " << std::setw(10) << shrink(info_collector.get_requests(), 1000) <<
				v2k(info_collector.get_requests(), 1000) << std::setprecision(5) <<
				" |  Total Entropy " << std::setw(10) << info_collector.get_entropy() <<
				" |  Local Entropy " << std::setw(10) << collector->get_entropy();
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

	ch_stream << "Total " << tcounter << endl <<
	"UB " << info_collector.get_unique_bytes() << endl <<
	"B " << info_collector.get_bytes() << endl <<
	"UR " << info_collector.get_unique_requests() << endl <<
	"R " << info_collector.get_requests() << endl <<
	"Entropy " << info_collector.get_entropy() << endl <<
	"Time " << latest_time - start_time;
	ch_stream.close();
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
