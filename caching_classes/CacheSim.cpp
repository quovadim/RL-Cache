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
    return byte_hits / (byte_hits + byte_misses);
}

uint64_t CacheSim::free_space() {
    return cache_size - used_space;
}

bool CacheSim::decide(p::dict request, double eviction_rating, int admission_decision) {
    prediction_updated_eviction = false;
    prediction_updated_admission = false;

    uint64_t size = p::extract<uint64_t>(request.get("size"));

    if (size > cache_size) {
	    misses += 1;
	    byte_misses += double(size) / 1000;
	    return false;
	}

	uint64_t id = p::extract<uint64_t>(request.get("id"));

    auto lit = cache.find(id);
	if (lit != cache.end()) {
		hits += 1;
		byte_hits += double(size) / 1000;

        double rating = L + eviction_rating;
        ratings.erase(lit->second);
		cache[id] = ratings.emplace(rating, id);
		prediction_updated_eviction = true;
		return true;
	}

	misses += 1;
	byte_misses += double(size) / 1000;

	prediction_updated_admission = true;

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

p::dict CacheSim::get_sizes() {
    return to_py_dict<uint64_t, uint64_t>(sizes);
}
void CacheSim::set_sizes(p::dict &_sizes) {
    sizes = to_std_map<uint64_t, uint64_t>(_sizes);
}

uint64_t CacheSim::get_used_space() {
    return used_space;
}
void CacheSim::set_used_space(uint64_t _used_space) {
    used_space = _used_space;
}

uint64_t CacheSim::get_cache_size() {
    return cache_size;
}
void CacheSim::set_cache_size(uint64_t _cache_size) {
    cache_size = _cache_size;
}

double CacheSim::get_L() {
    return L;
}
void CacheSim::set_L(double _L) {
    L = _L;
}

uint64_t CacheSim::get_misses() {
    return misses;
}
void CacheSim::set_misses(uint64_t _misses) {
    misses = _misses;
}

uint64_t CacheSim::get_hits() {
    return hits;
}
void CacheSim::set_hits(uint64_t _hits) {
    hits = _hits;
}

uint64_t CacheSim::get_byte_misses() {
    return byte_misses;
}
void CacheSim::set_byte_misses(uint64_t _byte_misses) {
    byte_misses = _byte_misses;
}

uint64_t CacheSim::get_byte_hits() {
    return byte_hits;
}
void CacheSim::set_byte_hits(uint64_t _byte_hits) {
    byte_hits = _byte_hits;
}

double CacheSim::get_total_rating() {
    return total_rating;
}
void CacheSim::set_total_rating(double _total_rating) {
    total_rating = _total_rating;
}

bool CacheSim::get_ml_eviction() {
    return is_ml_eviction;
}
void CacheSim::set_ml_eviction(double _is_ml_eviction) {
    is_ml_eviction = _is_ml_eviction;
}