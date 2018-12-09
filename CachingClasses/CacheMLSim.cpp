#include "CacheMLSim.h"
#include <math.h>
#include <iostream>
#include <unistd.h>

using std::cerr;
using std::endl;

CacheMLSimulator::CacheMLSimulator(uint64_t _cache_size, uint64_t _wing):
    CacheSim(_cache_size, 30),
    wing_size(_wing),
    last_dim(2 * _wing + 1)
{
    is_ml_eviction = true;
}

double CacheMLSimulator::sigmoid(double x) {
    return 1.0 / (1.0 + exp(x));
}

double CacheMLSimulator::convert_prediction_to_number(uint64_t prediction) {
	return pow(2.0, (double(prediction) - double(wing_size)));
}

double CacheMLSimulator::predict_eviction(p::list& eviction_features) {
    vector<double> prediction_features = vector<double>(to_std_vector<double>(eviction_features));
	if (!deterministic_eviction && (0.7 > distr(generator))) {
		latest_prediction_answer_eviction = sample(prediction_features);
		prediction_updated_eviction = true;
	} else {
		latest_prediction_answer_eviction = argmax(prediction_features);
	}

	return convert_prediction_to_number(latest_prediction_answer_eviction);
}

bool CacheMLSimulator::predict_admission(p::list& admission_features) {
    vector<double> prediction_features = vector<double>(to_std_vector<double>(admission_features));
	if (!deterministic_admission && (0.7 > distr(generator))) {
		latest_prediction_answer_admission = sample(prediction_features);
		prediction_updated_admission = true;
	} else {
		latest_prediction_answer_admission = argmax(prediction_features);
	}

	return latest_prediction_answer_admission == 1;
}

void CacheMLSimulator::produce_new_cache_state(p::dict &request, p::list& eviction_features, p::list& admission_features) {
	uint64_t size = p::extract<uint64_t>(request.get("size"));

	//if (!(predict_admission(admission_features))) {
    //    return;
	//}

	double prediction = predict_eviction(eviction_features);
	uint64_t id = p::extract<uint64_t>(request.get("id"));

    latest_mark[id] = prediction;

    double rating = L + prediction;
    total_rating += rating;
    cache[id] = ratings.emplace(rating, id);
	sizes.insert(pair<uint64_t, uint64_t>(id, size));
	updates.insert(pair<uint64_t, uint64_t>(id, 0));
	used_space += size;
	while (used_space > cache_size) {
	    auto min_elem = ratings.begin();
	    //L += pow(2.0, -1 * double(wing_size)) / 2.;
	    L = min_elem->first;
	    total_rating -= min_elem->first;
	    //misses_set.emplace(min_elem->second);
		used_space -= sizes[min_elem->second];
		cache.erase(min_elem->second);
		sizes.erase(min_elem->second);
		updates.erase(min_elem->second);
		latest_mark.erase(min_elem->second);
		ratings.erase(min_elem);
	}
	return;
}