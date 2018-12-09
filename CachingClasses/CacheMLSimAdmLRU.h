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


class CacheMLSimulatorAdmLRU : public CacheSim{
public:

	CacheMLSimulatorAdmLRU(uint64_t _cache_size, uint64_t _wing);

protected:
	uint64_t wing_size;
	uint64_t last_dim;

    virtual void produce_new_cache_state(p::dict &request, p::list& eviction_features, p::list& admission_features);

	virtual double predict_eviction(p::list& eviction_features);
	virtual bool predict_admission(p::list& admission_features);
};

BOOST_PYTHON_MODULE(CacheMLSimAdmLRU) {
	using namespace boost::python;

	class_<CacheMLSimulatorAdmLRU>("CacheMLSimulatorAdmLRU", init<uint64_t, uint64_t>())
	.def("__copy__", &generic__copy__< CacheMLSimulatorAdmLRU >)
    .def("__deepcopy__", &generic__deepcopy__< CacheMLSimulatorAdmLRU >)
	.def("reset", &CacheMLSimulatorAdmLRU::reset)
	.def("hit_rate", &CacheMLSimulatorAdmLRU::hit_rate)
	.def("byte_hit_rate", &CacheMLSimulatorAdmLRU::byte_hit_rate)
	.def("free_space", &CacheMLSimulatorAdmLRU::free_space)
	.def("decide", &CacheMLSimulatorAdmLRU::decide)
	.def("exp_hit_rate", &CacheMLSimulatorAdmLRU::exp_hit_rate)
	.def("get_admission_reward", &CacheMLSimulatorAdmLRU::get_admission_reward)
	.def("get_eviction_reward", &CacheMLSimulatorAdmLRU::get_eviction_reward)
	.def("get_ratings", &CacheMLSimulatorAdmLRU::get_ratings)
	.def("set_ratings", &CacheMLSimulatorAdmLRU::set_ratings)
	.def("get_sizes", &CacheMLSimulatorAdmLRU::get_sizes)
	.def("set_sizes", &CacheMLSimulatorAdmLRU::set_sizes)
	.def_readwrite("used_space", &CacheMLSimulatorAdmLRU::used_space)
	.def_readwrite("L", &CacheMLSimulatorAdmLRU::L)
	.def_readwrite("exponential_hit_rate", &CacheMLSimulatorAdmLRU::exponential_hit_rate)
	.def_readwrite("hits", &CacheMLSimulatorAdmLRU::hits)
	.def_readwrite("misses", &CacheMLSimulatorAdmLRU::misses)
	.def_readwrite("byte_hits", &CacheMLSimulatorAdmLRU::byte_hits)
	.def_readwrite("byte_misses", &CacheMLSimulatorAdmLRU::byte_misses)
	.def_readwrite("deterministic_eviction", &CacheMLSimulatorAdmLRU::deterministic_eviction)
	.def_readwrite("deterministic_admission", &CacheMLSimulatorAdmLRU::deterministic_admission)
	.def_readwrite("refresh_period", &CacheMLSimulatorAdmLRU::refresh_period)
	.def_readonly("prediction_updated_eviction", &CacheMLSimulatorAdmLRU::prediction_updated_eviction)
	.def_readonly("prediction_updated_admission", &CacheMLSimulatorAdmLRU::prediction_updated_admission)
	.def_readonly("latest_prediction_answer_eviction", &CacheMLSimulatorAdmLRU::latest_prediction_answer_eviction)
	.def_readonly("latest_prediction_answer_admission", &CacheMLSimulatorAdmLRU::latest_prediction_answer_admission);
}