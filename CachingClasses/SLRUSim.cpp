#include "SLRUSim.h"

SLRUSimulator::SLRUSimulator(uint64_t _cache_size) :
    CacheSim(_cache_size)
{
    for (uint64_t i = 0; i < 4; i++) {
        ids_distribution.push_back(vector<uint64_t>());
    }

    protected_size = cache_size / 2;
    public_size = cache_size / 2;
}

p::dict SLRUSimulator::get_cache() {
    cache.clear();
    for(auto it = protected_cache.begin(); it != protected_cache.end(); it++) {
        cache.insert(pair<uint64_t, double>(it->first, -1 * it->second));
    }
    for(auto it = public_cache.begin(); it != public_cache.end(); it++) {
        cache.insert(pair<uint64_t, double>(it->first, it->second));
    }
    return mm_to_py_dict(cache);
}

void SLRUSimulator::set_cache(p::dict& _cache) {
    cache = to_std_multimap<uint64_t, double>(_cache);
    for(auto it = cache.begin(); it != cache.end(); it++) {
        if (it->second < 0) {
            protected_cache.insert(pair<uint64_t, double>(it->first, -1 * it->second));
        } else {
            public_cache.insert(pair<uint64_t, double>(it->first, it->second))
        }
    }
}

double SLRUSimulator::predict_eviction(p::list& eviction_features) {
	prediction_updated_eviction = true;
	return vector<double>(to_std_vector<double>(eviction_features))[1];
}

bool SLRUSimulator::predict_admission(p::list& admission_features) {
	prediction_updated_admission = true;
	return to_std_vector<double>(admission_features)[2] > 0.5;
}

void SLRUSimulator::produce_new_cache_state(p::dict &request, p::list& eviction_features, p::list& admission_features) {
	if (!predict_admission(admission_features)) {
        return;
	}

	double prediction = predict_eviction(eviction_features);
	uint64_t id = p::extract<uint64_t>(request.get("id"));
	uint64_t size = p::extract<uint64_t>(request.get("size"));

	cache.insert(pair<uint64_t, double>(id, L + prediction));
	sizes.insert(pair<uint64_t, uint64_t>(id, size));
	used_space += size;
	while (used_space > cache_size) {
	    auto min_elem = min_element(cache.begin(), cache.end(), value_comparer);
		L = min_elem->second;
		used_space -= sizes.at(min_elem->first);
		sizes.erase(sizes.find(min_elem->first));
		cache.erase(cache.find(min_elem->first));
	}
	return;
}

bool SLRUSimulator::decide(p::dict request, p::list& eviction_features, p::list& admission_features) {
    prediction_updated_eviction = false;
    prediction_updated_admission = false;

    uint64_t size = p::extract<uint64_t>(request.get("size"));

    if (size > cache_size)
	    return false;

	uint64_t id = p::extract<uint64_t>(request.get("id"));
    auto lid = cache.find(id);
	if (lid != cache.end()) {
		hits += 1;
		byte_hits += size;
		exponential_hit_rate = WINDOW * exponential_hit_rate + 1 - WINDOW;
		uint64_t mask = 1 << (sizeof(T) * 8 - 1);
		if ()
		lid->second = L + uint64_t(predict_eviction(eviction_features));
		return true;
	}
	misses += 1;
	byte_misses += size;
	exponential_hit_rate = WINDOW * exponential_hit_rate;

	produce_new_cache_state(request, eviction_features, admission_features);
	return false;
}

map<uint64_t, double> SLRUSimulator::replace(map<uint64_t, double> &local_cache, map<uint64_t, double> replacement_objects) {
    map<uint64_t, double> erased_objects;

    uint64_t replacement_size = 0;
    for(auto it = replacement_objects.begin(); it != replacement_objects.end(); it++) {
        replacement_size += sizes.at(it->first);
    }
    for(auto it = replacement_objects.begin(); it != replacement_objects.end(); it++) {
        replacement_size += sizes.at(it->first);
    }

    return vector<uint64_t>();
}
map<uint64_t, double> SLRUSimulator::replace(map<uint64_t, double> &local_cache, uint64_t replacement_object, double rating) {
    map<uint64_t, double> tmp;
    tmp.insert(pair<uint64_t, double>(replacement_object, rating));
    return replace(local_cache, tmp);
}