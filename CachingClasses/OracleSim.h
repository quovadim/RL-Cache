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


class OracleSimulator : public CacheSim{
public:

	OracleSimulator(uint64_t _cache_size);

protected:
    virtual void produce_new_cache_state(p::dict &request, p::list& eviction_features, p::list& admission_features);

	virtual double predict_eviction(p::list& eviction_features);
	virtual bool predict_admission(p::list& admission_features);
};

BOOST_PYTHON_MODULE(OracleSim) {
	using namespace boost::python;

	class_<OracleSimulator>("OracleSimulator", init<uint64_t>())
	.def("__copy__", &generic__copy__< OracleSimulator >)
    .def("__deepcopy__", &generic__deepcopy__< OracleSimulator >)
	.def("reset", &OracleSimulator::reset)
	.def("hit_rate", &OracleSimulator::hit_rate)
	.def("byte_hit_rate", &OracleSimulator::byte_hit_rate)
	.def("free_space", &OracleSimulator::free_space)
	.def("decide", &OracleSimulator::decide)
	.def("exp_hit_rate", &OracleSimulator::exp_hit_rate)
	.def("get_admission_reward", &OracleSimulator::get_admission_reward)
	.def("get_eviction_reward", &OracleSimulator::get_eviction_reward)
		.def("get_ratings", &OracleSimulator::get_ratings)
	.def("set_ratings", &OracleSimulator::set_ratings)
	.def("get_sizes", &OracleSimulator::get_sizes)
	.def("set_sizes", &OracleSimulator::set_sizes)
	.def_readwrite("used_space", &OracleSimulator::used_space)
	.def_readwrite("L", &OracleSimulator::L)
	.def_readwrite("exponential_hit_rate", &OracleSimulator::exponential_hit_rate)
	.def_readwrite("hits", &OracleSimulator::hits)
	.def_readwrite("misses", &OracleSimulator::misses)
	.def_readwrite("byte_hits", &OracleSimulator::byte_hits)
	.def_readwrite("byte_misses", &OracleSimulator::byte_misses)
	.def_readwrite("deterministic_eviction", &OracleSimulator::deterministic_eviction)
	.def_readwrite("deterministic_admission", &OracleSimulator::deterministic_admission)
	.def_readwrite("refresh_period", &OracleSimulator::refresh_period)
	.def_readonly("prediction_updated_eviction", &OracleSimulator::prediction_updated_eviction)
	.def_readonly("prediction_updated_admission", &OracleSimulator::prediction_updated_admission)
	.def_readonly("latest_prediction_answer_eviction", &OracleSimulator::latest_prediction_answer_eviction)
	.def_readonly("latest_prediction_answer_admission", &OracleSimulator::latest_prediction_answer_admission);
}