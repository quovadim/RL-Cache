#include "SNLRUSim.h"
#include "auxiliary.hpp"

#include <iostream>

using std::cerr;
using std::endl;

SNLRUSimulator::SNLRUSimulator(uint64_t _cache_size, uint64_t _number_of_caches) :
    CacheSim(_cache_size)
{
    number_of_caches = _number_of_caches;
    segment_size = _cache_size / number_of_caches;
    for (uint64_t i = 0; i < number_of_caches; i++) {
        vector_cache.push_back(map<uint64_t, mm_iterator>());
	    vector_ratings.push_back(multimap<double, uint64_t>());
        vector_used_space.push_back(0);
        vector_L.push_back(0);
    }

    cache_size = segment_size * number_of_caches;
}

uint64_t SNLRUSimulator::free_space() {
    uint64_t free_space_amount = 0;
    for (uint64_t i = 0; i < number_of_caches; i++) {
        free_space_amount += segment_size - vector_used_space[i];
    }
    return free_space_amount;
}

uint64_t SNLRUSimulator::get_cache_index(uint64_t id) {
    for (uint64_t i = 0; i < number_of_caches; i++) {
        if (vector_cache[i].find(id) != vector_cache[i].end())
            return i;
    }
    return number_of_caches;
}

void SNLRUSimulator::admit_to_cache(vector<uint64_t> ids, vector<double> ratings, uint64_t cache_index) {
    for (uint64_t i = 0; i < ids.size(); i++) {
        uint64_t size = sizes[ids[i]];
        admit_to_cache(ids[i], size, ratings[i], cache_index);
    }
}

void SNLRUSimulator::admit_to_cache(uint64_t id, uint64_t size, double rating, uint64_t cache_index) {

    vector<uint64_t> erased_ids;
    vector<double> erased_ratings;

    while (vector_used_space[cache_index] + size > segment_size) {
	    auto min_elem = vector_ratings[cache_index].begin();

	    vector_used_space[cache_index] -= sizes[min_elem->second];

        if (cache_index != 0) {
            erased_ids.push_back(min_elem->second);
            erased_ratings.push_back(min_elem->first);
        }

        vector_cache[cache_index].erase(min_elem->second);
        vector_ratings[cache_index].erase(min_elem);

        if (cache_index == 0) {
            sizes.erase(min_elem->second);
		}
	}

    if (cache_index != 0) {
        admit_to_cache(erased_ids, erased_ratings, cache_index - 1);
    };

    total_rating += rating;
    vector_cache[cache_index][id] = vector_ratings[cache_index].emplace(rating, id);
	sizes.insert(pair<uint64_t, uint64_t>(id, size));
	vector_used_space[cache_index] += size;
}

bool SNLRUSimulator::decide(p::dict request, double eviction_rating, int admission_decision) {
    prediction_updated_eviction = false;
    prediction_updated_admission = false;

    uint64_t size = p::extract<uint64_t>(request.get("size"));

    if (size > segment_size) {
	    misses += 1;
	    byte_misses += double(size) / 1000;
	    return false;
	}

	uint64_t id = p::extract<uint64_t>(request.get("id"));

    uint64_t cache_index = get_cache_index(id);
	if (cache_index != number_of_caches) {
		hits += 1;
		byte_hits += double(size) / 1000;

        double rating = L + eviction_rating;

        auto lit = vector_cache[cache_index].find(id);

        if (cache_index + 1 != number_of_caches) {
            vector_ratings[cache_index].erase(lit->second);
            vector_cache[cache_index].erase(lit);
            vector_used_space[cache_index] -= size;

		    admit_to_cache(id, size, rating, cache_index + 1);

		} else {
            vector_ratings[cache_index].erase(lit->second);
		    vector_cache[cache_index][id] = vector_ratings[cache_index].emplace(rating, id);
		}

		prediction_updated_eviction = true;

		return true;
	}

	misses += 1;
	byte_misses += double(size) / 1000;

	prediction_updated_admission = true;

	produce_new_cache_state(request, eviction_rating, admission_decision);

	return false;
}


void SNLRUSimulator::produce_new_cache_state(p::dict &request, double eviction_rating, int admission_decision) {
	uint64_t size = p::extract<uint64_t>(request.get("size"));
	uint64_t id = p::extract<uint64_t>(request.get("id"));

	if (!admission_decision) {
        	return;
	}

	prediction_updated_eviction = true;

	admit_to_cache(id, size, eviction_rating, 0);

	return;
}

p::dict SNLRUSimulator::get_ratings() {
    map<uint64_t, p::dict> final_map;
    for (uint64_t i = 0; i < number_of_caches; i++) {
        map<uint64_t, double> local_map;
        for (auto it = vector_ratings[i].begin(); it != vector_ratings[i].end(); it++) {
            local_map[it->second] = it->first;
        }
        final_map[i] = to_py_dict(local_map);
    }
    return to_py_dict(final_map);
}
void SNLRUSimulator::set_ratings(p::dict &_ratings) {

    vector_cache = vector<map<uint64_t, mm_iterator>>();
    vector_ratings = vector<multimap<double, uint64_t>>();

    map<uint64_t, p::dict> final_map = to_std_map<uint64_t, p::dict>(_ratings);

    for (uint64_t i = 0; i < number_of_caches; i++) {
        vector_cache.push_back(map<uint64_t, mm_iterator>());
	    vector_ratings.push_back(multimap<double, uint64_t>());

        map<uint64_t, double> local_map = to_std_map<uint64_t, double>(final_map.at(i));
        for (auto it = local_map.begin(); it != local_map.end(); it++) {
            vector_cache[i][it->first] = vector_ratings[i].emplace(it->second, it->first);
        }
    }
}

