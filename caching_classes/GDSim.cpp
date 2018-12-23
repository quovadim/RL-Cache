#include "GDSim.h"
#include <math.h>
#include <iostream>
#include <unistd.h>

using std::cerr;
using std::endl;

GDSimulator::GDSimulator(uint64_t _cache_size):
    CacheSim(_cache_size)
{
    is_ml_eviction = true;
}

void GDSimulator::produce_new_cache_state(p::dict &request, double eviction_rating, bool admission_decision) {
	uint64_t size = p::extract<uint64_t>(request.get("size"));

	if (!admission_decision) {
        return;
	}

	if (size * 16 > cache_size) {
		return;
	}

	prediction_updated_eviction = true;

	while (used_space + size > cache_size) {
	    auto min_elem = ratings.begin();

	    L = min_elem->first;
	    used_space -= sizes[min_elem->second];

		cache.erase(min_elem->second);
		sizes.erase(min_elem->second);
		updates.erase(min_elem->second);
		latest_mark.erase(min_elem->second);
		ratings.erase(min_elem);
	}

	double prediction = eviction_rating;
	uint64_t id = p::extract<uint64_t>(request.get("id"));

    latest_mark[id] = prediction;

    double rating = L + prediction;
    total_rating += rating;
    cache[id] = ratings.emplace(rating, id);
	sizes.insert(pair<uint64_t, uint64_t>(id, size));
	updates.insert(pair<uint64_t, uint64_t>(id, 0));
	used_space += size;

	return;
}
