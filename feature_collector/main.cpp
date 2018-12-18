#include <iostream>
#include <vector>
#include <stdint.h>

#include <sstream>
#include <string>
#include <istream>
#include <fstream>
#include <iostream>
#include <stdlib.h>

#include "Environment.h"
#include "FeatureCollector.h"

using namespace std;
using std::cerr;
using std::endl;

#include <ctime>


int main(int argc, char** argv) {
	if (argc != 4) {
		cerr << "arguments : source_directory target_directory number_of_files" << endl;
		return -1;
	}
	string source_dir = argv[1];
	string target_dir = argv[2];
	uint64_t files = atoi(argv[3]);
	auto collector = new FeatureCollector();
	Environment env(collector, true);
	env.calculate_features(source_dir, target_dir, files);
    return 0;
}
