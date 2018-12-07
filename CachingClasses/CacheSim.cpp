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
	deterministic_eviction(false),
	deterministic_admission(false),
	prediction_updated_eviction(false),
	prediction_updated_admission(false),
	latest_prediction_answer_eviction(0),
	latest_prediction_answer_admission(0),
	used_space(0),
    L(0),
	exponential_hit_rate(0),
	hits(0),
	misses(0),
	byte_hits(0),
	byte_misses(0),
	admission_hits(0),
	admission_misses(0),
	cache_size(_cache_size),
	total_rating(0),
	eviction_hits_rating(0),
	eviction_misses_rating(0),
	eviction_byte_hits_rating(0),
	eviction_byte_misses_rating(0),
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

	exponential_hit_rate = 0;

	eviction_hits_rating = 0;
	eviction_misses_rating = 0;

	eviction_byte_hits_rating = 0;
	eviction_byte_misses_rating = 0;

	prediction_updated_eviction = false;
	prediction_updated_admission = false;

	latest_prediction_answer_eviction = 0;
	latest_prediction_answer_admission = 0;

    admission_hits = 0;
    admission_misses = 0;
	hits_set = unordered_set<uint64_t>();
	misses_set = unordered_set<uint64_t>();
}

double CacheSim::hit_rate() {
    return double(hits) / double(hits + misses);
}

double CacheSim::byte_hit_rate() {
    return double(byte_hits) / double(byte_hits + byte_misses);
}

double CacheSim::exp_hit_rate() {
    return exponential_hit_rate;
}

uint64_t CacheSim::free_space() {
    return cache_size - used_space;
}

double CacheSim::get_admission_reward() {
    return double(admission_hits) / (admission_hits + admission_misses);
}

int64_t CacheSim::get_eviction_reward() {
    return 0;
}

double CacheSim::eviction_rating() {
    return eviction_hits_rating / (eviction_hits_rating + eviction_misses_rating);
}

double CacheSim::byte_eviction_rating() {
    return eviction_byte_hits_rating / (eviction_byte_hits_rating + eviction_byte_misses_rating);
}

bool CacheSim::decide(p::dict request, p::list& eviction_features, p::list& admission_features) {
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
		exponential_hit_rate = WINDOW * exponential_hit_rate + 1 - WINDOW;

        double old_rating = lit->second->first;
        double estimation = predict_eviction(eviction_features);

        //if (!deterministic_eviction && is_ml_eviction && (distr(generator) < 0.2)) {
        // (old_rating > (L + estimation)) FOR MAX SELECTION prediction_updated_eviction = false
        // (distr(generator) < std::pow(2, -1 * std::fabs(std::log2(latest_mark[id]) - std::log2(estimation)))) FOR RANDOM
        if (is_ml_eviction && (old_rating > (L + estimation))) {
            ratings.erase(lit->second);
		    cache[id] = ratings.emplace(L + latest_mark[id], id);
            prediction_updated_eviction = false;
            //latest_prediction_answer_eviction = uint64_t(std::fabs(std::log2(latest_mark[id])));
            return true;
        }

        double rating = 0;

        latest_mark[id] = estimation;

        if (is_ml_eviction) {
            rating = L + estimation;
        } else {
            rating = L + estimation;
        }

        ratings.erase(lit->second);
		cache[id] = ratings.emplace(rating, id);
        //if (rating <= old_rating) {
        //    prediction_updated_eviction = false;
        //    return true;
        //}

        double max_rating = max(old_rating, ratings.rbegin()->first);
		double min_rating = min(old_rating, ratings.begin()->first);
		double rating_score = 0;
        if (max_rating - min_rating > 1e-6) {
            rating_score = 0.5 + 0.5 * (old_rating - min_rating) / (max_rating - min_rating);

        }
        //    double diff_rating = 1 - (rating - old_rating) / (max_rating - min_rating);
        //    rating_score = diff_rating * ((max_rating - old_rating) / (max_rating - min_rating));
		//}
		eviction_hits_rating += rating_score;
		eviction_byte_hits_rating += size * rating_score;
		//total_rating -= old_rating;
		//total_rating += rating;
		return true;
	}

	misses += 1;
	byte_misses += size;
	eviction_misses_rating += 1;
	eviction_byte_misses_rating += size;
	exponential_hit_rate = WINDOW * exponential_hit_rate;

	produce_new_cache_state(request, eviction_features, admission_features);
	return false;
}

uint64_t CacheSim::argmax(vector<double> data) {
    double cmax = 0;
    uint64_t index_max = 0;
	for(uint64_t i = 0; i < data.size(); i++) {
		if (cmax < data[i]) {
			cmax = data[i];
			index_max = i;
		}
	}
	return index_max;
}

uint64_t CacheSim::sample(std::vector<double> distribution) {
	double target = distr(generator);
	double summary = 0;
	double total = 0;
		for (uint64_t i = 0; i < distribution.size(); i++) {
		total += distribution[i];
	}
	target *= total;
	for (uint64_t i = 0; i < distribution.size(); i++) {
		summary += distribution[i];
		if (summary >= target) {
			return i;
		}
	}
	return distribution.size() - 1;
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
    return to_py_dict(sizes);
}
void CacheSim::set_sizes(p::dict &_sizes) {
    sizes = to_std_map<uint64_t, uint64_t>(_sizes);
}