#include "CacheSim.h"
#include "auxiliary.hpp"

#include <map>
#include <unordered_set>
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <math.h>

using std::map;
using std::vector;
using std::cerr;
using std::endl;
using std::multimap;
using std::unordered_set;
using std::max;
using std::min;

namespace p = boost::python;

CacheSim::CacheSim(uint64_t _cache_size):
	prediction_updated_eviction(false),
	prediction_updated_admission(false),
	refresh_period(0),
	used_space(0),
    L(0),
	hits(0),
	misses(0),
	byte_hits(0),
	byte_misses(0),
	cache_size(_cache_size),
	is_ml_eviction(false)
{
    std::random_device device;
	generator = std::mt19937(device());
	distr = std::uniform_real_distribution<>(0, 1);
}


void CacheSim::reset() {

    std::random_device device;
	generator = std::mt19937(device());
	distr = std::uniform_real_distribution<>(0, 1);

	hits = 0;
	misses = 0;

    byte_hits = 0;
	byte_misses = 0;

}

double CacheSim::hit_rate() {
    return double(hits) / double(hits + misses);
}

double CacheSim::byte_hit_rate() {
    return double(byte_hits) / double(byte_hits + byte_misses);
}

uint64_t CacheSim::free_space() {
    return cache_size - used_space;
}

bool CacheSim::decide(p::dict request, double eviction_rating, bool admission_decision) {
    prediction_updated_eviction = false;
    prediction_updated_admission = false;

    uint64_t size = p::extract<uint64_t>(request.get("size"));

    if (size > cache_size)
	    return false;

	uint64_t id = p::extract<uint64_t>(request.get("id"));

    auto lit = cache.find(id);
	if (lit != cache.end()) {
		hits += 1;
		byte_hits += size;

		if (is_ml_eviction) {
		    if (updates[id] < refresh_period) {
                updates[id]++;
                ratings.erase(lit->second);
		        cache[id] = ratings.emplace(L + latest_mark[id], id);
                prediction_updated_eviction = false;
                return true;
            } else {
                updates[id] = 0;
            }
        }

        //if (!deterministic_eviction && is_ml_eviction && (distr(generator) < 0.2)) {
        // (old_rating > (L + estimation)) FOR MAX SELECTION prediction_updated_eviction = false
        // (distr(generator) < std::pow(2, -1 * std::fabs(std::log2(latest_mark[id]) - std::log2(estimation)))) FOR RANDOM
        //if (is_ml_eviction && ((L - (old_rating - latest_mark[id])) < 128)) {

        double rating = L + eviction_rating;
        latest_mark[id] = eviction_rating;
        ratings.erase(lit->second);
		cache[id] = ratings.emplace(rating, id);
		prediction_updated_eviction = true;
		return true;
	}

	misses += 1;
	byte_misses += size;

	prediction_updated_admission = true;
	prediction_updated_eviction = true;

	produce_new_cache_state(request, eviction_rating, admission_decision);

	return false;
}

p::dict CacheSim::get_ratings() {
    map<uint64_t, double> final_map;
    for(auto it = ratings.begin(); it != ratings.end(); it++) {
        final_map[it->second] = it->first;
    }
    return to_py_dict(final_map);
}
void CacheSim::set_ratings(p::dict &_ratings) {
    cache = map<uint64_t, mm_iterator>();
    ratings = multimap<double, uint64_t>();
    map<uint64_t, double> final_map = to_std_map<uint64_t, double>(_ratings);
    for(auto it = final_map.begin(); it != final_map.end(); it++) {
        cache[it->first] = ratings.emplace(it->second, it->first);
    }
}