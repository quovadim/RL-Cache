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


class CacheMLSimulator : public CacheSim{
public:

	CacheMLSimulator(uint64_t _cache_size, uint64_t _wing);

protected:
	uint64_t wing_size;
	uint64_t last_dim;

    double sigmoid(double x);
	double convert_prediction_to_number(uint64_t prediction);

    virtual void produce_new_cache_state(p::dict &request, p::list& eviction_features, p::list& admission_features);

	virtual double predict_eviction(p::list& eviction_features);
	virtual bool predict_admission(p::list& admission_features);
};

BOOST_PYTHON_MODULE(CacheMLSim) {
	using namespace boost::python;

	class_<CacheMLSimulator>("CacheMLSimulator", init<uint64_t, uint64_t>())
	.def("__copy__", &generic__copy__< CacheMLSimulator >)
    .def("__deepcopy__", &generic__deepcopy__< CacheMLSimulator >)
	.def("reset", &CacheMLSimulator::reset)
	.def("hit_rate", &CacheMLSimulator::hit_rate)
	.def("byte_hit_rate", &CacheMLSimulator::byte_hit_rate)
	.def("free_space", &CacheMLSimulator::free_space)
	.def("decide", &CacheMLSimulator::decide)
	.def("exp_hit_rate", &CacheMLSimulator::exp_hit_rate)
	.def("get_admission_reward", &CacheMLSimulator::get_admission_reward)
	.def("get_eviction_reward", &CacheMLSimulator::get_eviction_reward)
	.def("get_ratings", &CacheMLSimulator::get_ratings)
	.def("set_ratings", &CacheMLSimulator::set_ratings)
	.def("get_sizes", &CacheMLSimulator::get_sizes)
	.def("set_sizes", &CacheMLSimulator::set_sizes)
	.def("eviction_rating", &CacheMLSimulator::eviction_rating)
	.def("byte_eviction_rating", &CacheMLSimulator::byte_eviction_rating)
	.def_readwrite("used_space", &CacheMLSimulator::used_space)
	.def_readwrite("L", &CacheMLSimulator::L)
	.def_readwrite("exponential_hit_rate", &CacheMLSimulator::exponential_hit_rate)
	.def_readwrite("hits", &CacheMLSimulator::hits)
	.def_readwrite("misses", &CacheMLSimulator::misses)
	.def_readwrite("byte_hits", &CacheMLSimulator::byte_hits)
	.def_readwrite("byte_misses", &CacheMLSimulator::byte_misses)
	.def_readwrite("deterministic_eviction", &CacheMLSimulator::deterministic_eviction)
	.def_readwrite("deterministic_admission", &CacheMLSimulator::deterministic_admission)
	.def_readonly("prediction_updated_eviction", &CacheMLSimulator::prediction_updated_eviction)
	.def_readonly("prediction_updated_admission", &CacheMLSimulator::prediction_updated_admission)
	.def_readonly("latest_prediction_answer_eviction", &CacheMLSimulator::latest_prediction_answer_eviction)
	.def_readonly("latest_prediction_answer_admission", &CacheMLSimulator::latest_prediction_answer_admission);
}