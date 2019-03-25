#pragma once

#include <Python.h>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <map>
#include <vector>
#include <math.h>
#include <random>
#include <algorithm>

#include "auxiliary.hpp"
#include "CacheSim.h"

using std::map;
using std::vector;
using std::exp;
using std::pair;

namespace p = boost::python;

typedef map<uint64_t, double> dict_predictions;
typedef map<uint64_t, uint64_t> dict_sizes;

class SNLRUSimulator : public CacheSim {
public:

	SNLRUSimulator(uint64_t _cache_size, uint64_t _number_of_caches);

	virtual uint64_t free_space();

	virtual bool decide(p::dict request, double eviction_rating, int admission_decision);

	virtual p::dict get_ratings();
	virtual void set_ratings(p::dict &_ratings);

protected:
    virtual void produce_new_cache_state(p::dict &request, double eviction_rating, int admission_rating);

    virtual void admit_to_cache(vector<uint64_t> ids, vector<double> ratings, uint64_t cache_index);
    virtual void admit_to_cache(uint64_t id, uint64_t size, double rating, uint64_t cache_index);
    virtual uint64_t get_cache_index(uint64_t id);

private:

    uint64_t number_of_caches;
    uint64_t segment_size;
    vector<map<uint64_t, mm_iterator>> vector_cache;
	vector<multimap<double, uint64_t>> vector_ratings;
	map<uint64_t, uint64_t> sizes;

    vector<uint64_t> vector_used_space;

    vector<double> vector_L;

	vector<uint64_t> vector_cache_size;

	double total_rating;

	bool is_ml_eviction;
};