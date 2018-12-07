#include "CacheMLSimAdm.h"
#include <iostream>

using std::cerr;
using std::endl;

CacheMLSimulatorAdm::CacheMLSimulatorAdm(uint64_t _cache_size, uint64_t _wing) :
    CacheSim(_cache_size),
    wing_size(_wing),
    last_dim(2 * _wing + 1)
{}

double CacheMLSimulatorAdm::predict_eviction(p::list& eviction_features) {
    prediction_updated_eviction = true;
    return vector<double>(to_std_vector<double>(eviction_features))[0];
}

bool CacheMLSimulatorAdm::predict_admission(p::list& admission_features) {
    vector<double> prediction_features = vector<double>(to_std_vector<double>(admission_features));
	if (!deterministic_admission) {
		latest_prediction_answer_admission = sample(prediction_features);
	} else {
		latest_prediction_answer_admission = argmax(prediction_features);
	}
	prediction_updated_admission = true;

	return latest_prediction_answer_admission == 1;
}

void CacheMLSimulatorAdm::produce_new_cache_state(p::dict &request, p::list& eviction_features, p::list& admission_features) {
	uint64_t size = p::extract<uint64_t>(request.get("size"));

	if (!(predict_admission(admission_features))) {
        return;
	}

	double prediction = predict_eviction(eviction_features);
	uint64_t id = p::extract<uint64_t>(request.get("id"));

	latest_mark[id] = prediction;

    double rating = L + prediction;
    total_rating += rating;
    cache[id] = ratings.emplace(rating, id);
	sizes.insert(pair<uint64_t, uint64_t>(id, size));
	used_space += size;
	while (used_space > cache_size) {
	    auto min_elem = ratings.begin();
	    L = min_elem->first;
	    total_rating -= min_elem->first;
	    misses_set.emplace(min_elem->second);
		used_space -= sizes[min_elem->second];
		cache.erase(min_elem->second);
		sizes.erase(min_elem->second);
		latest_mark.erase(min_elem->second);
		ratings.erase(min_elem);
	}
	return;
}