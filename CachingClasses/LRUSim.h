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


class LRUSimulator : public CacheSim{
public:

	LRUSimulator(uint64_t _cache_size);

protected:
    virtual void produce_new_cache_state(p::dict &request, p::list& eviction_features, p::list& admission_features);

	virtual double predict_eviction(p::list& eviction_features);
	virtual bool predict_admission(p::list& admission_features);
};

BOOST_PYTHON_MODULE(LRUSim) {
	using namespace boost::python;

	class_<LRUSimulator>("LRUSimulator", init<uint64_t>())
	.def("__copy__", &generic__copy__< LRUSimulator >)
    .def("__deepcopy__", &generic__deepcopy__< LRUSimulator >)
	.def("reset", &LRUSimulator::reset)
	.def("hit_rate", &LRUSimulator::hit_rate)
	.def("byte_hit_rate", &LRUSimulator::byte_hit_rate)
	.def("free_space", &LRUSimulator::free_space)
	.def("decide", &LRUSimulator::decide)
	.def("exp_hit_rate", &LRUSimulator::exp_hit_rate)
	.def("get_admission_reward", &LRUSimulator::get_admission_reward)
	.def("get_eviction_reward", &LRUSimulator::get_eviction_reward)
    .def("get_ratings", &LRUSimulator::get_ratings)
	.def("set_ratings", &LRUSimulator::set_ratings)
	.def("get_sizes", &LRUSimulator::get_sizes)
	.def("set_sizes", &LRUSimulator::set_sizes)
	.def_readwrite("used_space", &LRUSimulator::used_space)
	.def_readwrite("L", &LRUSimulator::L)
	.def_readwrite("exponential_hit_rate", &LRUSimulator::exponential_hit_rate)
	.def_readwrite("hits", &LRUSimulator::hits)
	.def_readwrite("misses", &LRUSimulator::misses)
	.def_readwrite("byte_hits", &LRUSimulator::byte_hits)
	.def_readwrite("byte_misses", &LRUSimulator::byte_misses)
	.def_readwrite("deterministic_eviction", &LRUSimulator::deterministic_eviction)
	.def_readwrite("deterministic_admission", &LRUSimulator::deterministic_admission)
	.def_readonly("prediction_updated_eviction", &LRUSimulator::prediction_updated_eviction)
	.def_readonly("prediction_updated_admission", &LRUSimulator::prediction_updated_admission)
	.def_readonly("latest_prediction_answer_eviction", &LRUSimulator::latest_prediction_answer_eviction)
	.def_readonly("latest_prediction_answer_admission", &LRUSimulator::latest_prediction_answer_admission);
}