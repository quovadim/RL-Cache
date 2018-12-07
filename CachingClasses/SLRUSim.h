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


class SLRUSimulator : public CacheSim{
public:

	SLRUSimulator(uint64_t _cache_size);

protected:

    multimap<uint64_t, double> protected_cache;
    multimap<uint64_t, double> public_cache;

    uint64_t protected_size;
    uint64_t public_size;

    virtual void produce_new_cache_state(p::dict &request, p::list& eviction_features, p::list& admission_features);

	virtual double predict_eviction(p::list& eviction_features);
	virtual bool predict_admission(p::list& admission_features);

    virtual bool decide(p::dict request, p::list& eviction_features, p::list& admission_features);

    map<uint64_t, double> replace(map<uint64_t, double> &local_cache, map<uint64_t, double> replacement_objects);
    map<uint64_t, double> replace(map<uint64_t, double> &local_cache, uint64_t replacement_object, double rating);

	vector<vector<uint64_t>> ids_distribution;
};

BOOST_PYTHON_MODULE(LRUSim) {
	using namespace boost::python;

	class_<SLRUSimulator>("S4LRUSimulator", init<uint64_t>())
	.def("reset", &SLRUSimulator::reset)
	.def("hit_rate", &SLRUSimulator::hit_rate)
	.def("byte_hit_rate", &SLRUSimulator::byte_hit_rate)
	.def("free_space", &SLRUSimulator::free_space)
	.def("decide", &SLRUSimulator::decide)
	.def("exp_hit_rate", &SLRUSimulator::exp_hit_rate)
	.def("get_cache", &SLRUSimulator::get_cache)
	.def("set_cache", &SLRUSimulator::set_cache)
	.def("get_sizes", &SLRUSimulator::get_sizes)
    .def("set_sizes", &SLRUSimulator::set_sizes)
	.def_readwrite("deterministic_eviction", &SLRUSimulator::deterministic_eviction)
	.def_readwrite("deterministic_admission", &SLRUSimulator::deterministic_admission)
	.def_readwrite("used_space", &SLRUSimulator::used_space)
	.def_readwrite("L", &SLRUSimulator::L)
	.def_readwrite("exponential_hit_rate", &SLRUSimulator::exponential_hit_rate)
	.def_readwrite("hits", &SLRUSimulator::hits)
	.def_readwrite("misses", &SLRUSimulator::misses)
	.def_readwrite("byte_hits", &SLRUSimulator::byte_hits)
	.def_readwrite("byte_misses", &SLRUSimulator::byte_misses)
	.def_readonly("prediction_updated_eviction", &SLRUSimulator::prediction_updated_eviction)
	.def_readonly("prediction_updated_admission", &SLRUSimulator::prediction_updated_admission)
	.def_readonly("latest_prediction_answer_eviction", &SLRUSimulator::latest_prediction_answer_eviction)
	.def_readonly("latest_prediction_answer_admission", &SLRUSimulator::latest_prediction_answer_admission);
}