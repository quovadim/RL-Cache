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


class GDSFSimulator : public CacheSim{
public:

	GDSFSimulator(uint64_t _cache_size);

protected:
    virtual void produce_new_cache_state(p::dict &request, p::list& eviction_features, p::list& admission_features);

	virtual double predict_eviction(p::list& eviction_features);
	virtual bool predict_admission(p::list& admission_features);
};

BOOST_PYTHON_MODULE(GDSFSim) {
	using namespace boost::python;

	class_<GDSFSimulator>("GDSFSimulator", init<uint64_t>())
	.def("__copy__", &generic__copy__< GDSFSimulator >)
    .def("__deepcopy__", &generic__deepcopy__< GDSFSimulator >)
	.def("reset", &GDSFSimulator::reset)
	.def("hit_rate", &GDSFSimulator::hit_rate)
	.def("byte_hit_rate", &GDSFSimulator::byte_hit_rate)
	.def("free_space", &GDSFSimulator::free_space)
	.def("decide", &GDSFSimulator::decide)
	.def("exp_hit_rate", &GDSFSimulator::exp_hit_rate)
	.def("get_admission_reward", &GDSFSimulator::get_admission_reward)
	.def("get_eviction_reward", &GDSFSimulator::get_eviction_reward)
		.def("get_ratings", &GDSFSimulator::get_ratings)
	.def("set_ratings", &GDSFSimulator::set_ratings)
	.def("get_sizes", &GDSFSimulator::get_sizes)
	.def("set_sizes", &GDSFSimulator::set_sizes)
	.def_readwrite("used_space", &GDSFSimulator::used_space)
	.def_readwrite("L", &GDSFSimulator::L)
	.def_readwrite("exponential_hit_rate", &GDSFSimulator::exponential_hit_rate)
	.def_readwrite("hits", &GDSFSimulator::hits)
	.def_readwrite("misses", &GDSFSimulator::misses)
	.def_readwrite("byte_hits", &GDSFSimulator::byte_hits)
	.def_readwrite("byte_misses", &GDSFSimulator::byte_misses)
	.def_readwrite("deterministic_eviction", &GDSFSimulator::deterministic_eviction)
	.def_readwrite("deterministic_admission", &GDSFSimulator::deterministic_admission)
	.def_readonly("prediction_updated_eviction", &GDSFSimulator::prediction_updated_eviction)
	.def_readonly("prediction_updated_admission", &GDSFSimulator::prediction_updated_admission)
	.def_readonly("latest_prediction_answer_eviction", &GDSFSimulator::latest_prediction_answer_eviction)
	.def_readonly("latest_prediction_answer_admission", &GDSFSimulator::latest_prediction_answer_admission);
}