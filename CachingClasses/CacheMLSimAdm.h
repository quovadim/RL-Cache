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


class CacheMLSimulatorAdm : public CacheSim{
public:

	CacheMLSimulatorAdm(uint64_t _cache_size, uint64_t _wing);

protected:
	uint64_t wing_size;
	uint64_t last_dim;

    virtual void produce_new_cache_state(p::dict &request, p::list& eviction_features, p::list& admission_features);

	virtual double predict_eviction(p::list& eviction_features);
	virtual bool predict_admission(p::list& admission_features);
};

BOOST_PYTHON_MODULE(CacheMLSimAdm) {
	using namespace boost::python;

	class_<CacheMLSimulatorAdm>("CacheMLSimulatorAdm", init<uint64_t, uint64_t>())
	.def("__copy__", &generic__copy__< CacheMLSimulatorAdm >)
    .def("__deepcopy__", &generic__deepcopy__< CacheMLSimulatorAdm >)
	.def("reset", &CacheMLSimulatorAdm::reset)
	.def("hit_rate", &CacheMLSimulatorAdm::hit_rate)
	.def("byte_hit_rate", &CacheMLSimulatorAdm::byte_hit_rate)
	.def("free_space", &CacheMLSimulatorAdm::free_space)
	.def("decide", &CacheMLSimulatorAdm::decide)
	.def("exp_hit_rate", &CacheMLSimulatorAdm::exp_hit_rate)
	.def("get_admission_reward", &CacheMLSimulatorAdm::get_admission_reward)
	.def("get_eviction_reward", &CacheMLSimulatorAdm::get_eviction_reward)
	.def("get_ratings", &CacheMLSimulatorAdm::get_ratings)
	.def("set_ratings", &CacheMLSimulatorAdm::set_ratings)
	.def("get_sizes", &CacheMLSimulatorAdm::get_sizes)
	.def("set_sizes", &CacheMLSimulatorAdm::set_sizes)
	.def_readwrite("used_space", &CacheMLSimulatorAdm::used_space)
	.def_readwrite("L", &CacheMLSimulatorAdm::L)
	.def_readwrite("exponential_hit_rate", &CacheMLSimulatorAdm::exponential_hit_rate)
	.def_readwrite("hits", &CacheMLSimulatorAdm::hits)
	.def_readwrite("misses", &CacheMLSimulatorAdm::misses)
	.def_readwrite("byte_hits", &CacheMLSimulatorAdm::byte_hits)
	.def_readwrite("byte_misses", &CacheMLSimulatorAdm::byte_misses)
	.def_readwrite("deterministic_eviction", &CacheMLSimulatorAdm::deterministic_eviction)
	.def_readwrite("deterministic_admission", &CacheMLSimulatorAdm::deterministic_admission)
	.def_readonly("prediction_updated_eviction", &CacheMLSimulatorAdm::prediction_updated_eviction)
	.def_readonly("prediction_updated_admission", &CacheMLSimulatorAdm::prediction_updated_admission)
	.def_readonly("latest_prediction_answer_eviction", &CacheMLSimulatorAdm::latest_prediction_answer_eviction)
	.def_readonly("latest_prediction_answer_admission", &CacheMLSimulatorAdm::latest_prediction_answer_admission);
}