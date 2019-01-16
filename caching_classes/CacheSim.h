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

	virtual uint64_t free_space();

	virtual bool decide(p::dict request, double eviction_rating, bool admission_decision);

	bool prediction_updated_eviction;
	bool prediction_updated_admission;

	uint64_t refresh_period;

	p::dict get_ratings();
	void set_ratings(p::dict &_ratings);

	p::dict get_latest_marks();
	void set_latest_marks(p::dict &_latest_mark);

	p::dict get_updates();
	void set_updates(p::dict &_updates);

	p::dict get_sizes();
	void set_sizes(p::dict &_sizes);

	uint64_t get_used_space();
	void set_used_space(uint64_t _used_space);

	uint64_t get_cache_size();
	void set_cache_size(uint64_t _cache_size);

	double get_L();
	void set_L(double _L);

	uint64_t get_misses();
	void set_misses(uint64_t _misses);

	uint64_t get_hits();
	void set_hits(uint64_t _hits);

	uint64_t get_byte_misses();
	void set_byte_misses(uint64_t _byte_misses);

	uint64_t get_byte_hits();
	void set_byte_hits(uint64_t _byte_hits);

	double get_total_rating();
	void set_total_rating(double _total_rating);

	bool get_ml_eviction();
	void set_ml_eviction(double _is_ml_eviction);

protected:

    virtual void produce_new_cache_state(p::dict &request, double eviction_rating, bool admission_decision) = 0;

    map<uint64_t, mm_iterator> cache;
	multimap<double, uint64_t> ratings;
	map<uint64_t, uint64_t> sizes;
	map<uint64_t, double> latest_mark;
	map<uint64_t, uint64_t> updates;

	uint64_t used_space;

    double L;

	uint64_t hits;
	uint64_t misses;

	double byte_hits;
	double byte_misses;

	uint64_t cache_size;

	double total_rating;

	bool is_ml_eviction;

	std::mt19937 generator;
	std::uniform_real_distribution<> distr;
};