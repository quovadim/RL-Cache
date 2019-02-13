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


void Environment::calculate_features(string& prefix_input, string& prefix_output, uint64_t files) {
	for (int i = files - 1; i >= 0; i--) {
		vector<Packet*> dataset;
		dataset = iterate_csv(prefix_input + to_string(i) + ".csv");
		ofstream stream;
		stream.open(prefix_output + to_string(i) + ".csv");
		for (int j = dataset.size() - 1; j >= 0; j--) {
			Packet* current_packet = dataset[j];

			current_packet->rating = collector->get_packet_rating(current_packet);

			if (j % 1687 == 0) {
				cerr << std::fixed << std::setprecision(1) <<
				"\rFile " << to_string(i) + ".csv" <<
				" |  Local " << std::setw(6) << shrink(j, 1000) << v2k(j, 1000) <<
				" |  Size " << std::setw(6) << shrink(dataset.size(), 1000) << v2k(dataset.size(), 1000);
			}
        }

        for (uint64_t j = 0; j < dataset.size(); j++) {
			Packet* current_packet = dataset[j];

			stream << current_packet->timestamp << " "
			<< current_packet->id << " "
			<< current_packet->size << " "
			<< current_packet->l1 << " "
			<< current_packet->l2 << " "
			<< current_packet->l3 << " "
			<< current_packet->l4 << " "
			<< current_packet->l5 << " "
			<< current_packet->l6 << " "
			<< current_packet->rating << endl;

			if (j % 1687 == 0) {
				cerr << std::fixed << std::setprecision(1) <<
				"\rFile " << to_string(i) + ".csv" <<
				" |  Local " << std::setw(6) << shrink(j, 1000) << v2k(j, 1000) <<
				" |  Size " << std::setw(6) << shrink(dataset.size(), 1000) << v2k(dataset.size(), 1000);
			}

			delete current_packet;
		}

		cerr << endl;
		stream.close();
	}
}

vector<Packet*> Environment::iterate_csv(string filename) {
    ifstream input(filename);
    vector<Packet*> result;
    uint64_t time, id, size, l1, l2, l3;
    double l4, l5, l6;
    while(input >> time >> id >> size >> l1 >> l2 >> l3 >> l4 >> l5 >> l6) {
        Packet* new_packet = new Packet(id, size, time, l1, l2, l3, l4, l5, l6);
        result.emplace_back(new_packet);
    }
    return result;
}
