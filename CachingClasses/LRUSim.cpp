#include "LRUSim.h"

LRUSimulator::LRUSimulator(uint64_t _cache_size) :
    CacheSim(_cache_size, 0)
{}

double LRUSimulator::predict_eviction(p::list& eviction_features) {
	prediction_updated_eviction = true;
	return vector<double>(to_std_vector<double>(eviction_features))[1];
}

bool LRUSimulator::predict_admission(p::list& admission_features) {
	prediction_updated_admission = true;
	return to_std_vector<double>(admission_features)[2] > 0.5;
}

void LRUSimulator::produce_new_cache_state(p::dict &request, p::list& eviction_features, p::list& admission_features) {
	uint64_t size = p::extract<uint64_t>(request.get("size"));

	//if (!(predict_admission(admission_features))) {
    //    return;
	//}

	double prediction = predict_eviction(eviction_features);
	uint64_t id = p::extract<uint64_t>(request.get("id"));

	latest_mark[id] = prediction;

    double rating = prediction;
    cache[id] = ratings.emplace(pair<double, uint64_t>(rating, id));
	sizes.insert(pair<uint64_t, uint64_t>(id, size));
	used_space += size;
	while (used_space > cache_size) {
	    auto min_elem = ratings.begin();
	    misses_set.emplace(min_elem->second);
		used_space -= sizes[min_elem->second];
		cache.erase(min_elem->second);
		sizes.erase(min_elem->second);
		latest_mark.erase(min_elem->second);
		ratings.erase(min_elem);
	}
	return;
}