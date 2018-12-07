#include <Python.h>
#include <map>
#include <unordered_set>
#include <vector>
#include <random>
#include <algorithm>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>

using std::map;
using std::multimap;
using std::vector;
using std::unordered_set;

typedef multimap<double, uint64_t>::iterator mm_iterator;

namespace p = boost::python;

class CacheSim {
public:
    CacheSim(uint64_t cache_size);

	virtual void reset();

	virtual double hit_rate();

    virtual double byte_hit_rate();

    virtual double exp_hit_rate();

    virtual double get_admission_reward();

    virtual int64_t get_eviction_reward();

	virtual uint64_t free_space();

	virtual bool decide(p::dict request, p::list& eviction_features, p::list& admission_features);

    virtual double eviction_rating();
    virtual double byte_eviction_rating();

	bool deterministic_eviction;
	bool deterministic_admission;

	bool prediction_updated_eviction;
	bool prediction_updated_admission;

	uint64_t latest_prediction_answer_eviction;
	uint64_t latest_prediction_answer_admission;

	p::dict get_ratings();
	void set_ratings(p::dict &_ratings);

	p::dict get_sizes();
	void set_sizes(p::dict &_sizes);

	uint64_t used_space;

    double L;

	double exponential_hit_rate;

	uint64_t hits;
	uint64_t misses;

	uint64_t byte_hits;
	uint64_t byte_misses;

protected:

    virtual void produce_new_cache_state(p::dict &request, p::list& eviction_features, p::list& admission_features) = 0;

	virtual double predict_eviction(p::list& eviction_features) = 0;
	virtual bool predict_admission(p::list& admission_features) = 0;

    uint64_t argmax(vector<double> data);
	uint64_t sample(std::vector<double> distribution);

    map<uint64_t, mm_iterator> cache;
	multimap<double, uint64_t> ratings;
	map<uint64_t, uint64_t> sizes;
	map<uint64_t, double> latest_mark;

	unordered_set<uint64_t> hits_set;
	unordered_set<uint64_t> misses_set;
	uint64_t admission_hits;
	uint64_t admission_misses;

	uint64_t cache_size;

	double total_rating;

	double eviction_hits_rating;
	double eviction_misses_rating;

	double eviction_byte_hits_rating;
	double eviction_byte_misses_rating;

	bool is_ml_eviction;

	std::mt19937 generator;
	std::uniform_real_distribution<> distr;
};