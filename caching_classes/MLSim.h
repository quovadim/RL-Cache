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


class MLSimulator : public CacheSim {
public:
	MLSimulator(uint64_t _cache_size);

protected:
    virtual void produce_new_cache_state(p::dict &request, double eviction_rating, bool admission_decision);
};

BOOST_PYTHON_MODULE(MLSim) {
	using namespace boost::python;

	class_<MLSimulator>("MLSimulator", init<uint64_t>())
	.def("__copy__", &generic__copy__< MLSimulator >)
    .def("__deepcopy__", &generic__deepcopy__< MLSimulator >)
	.def("reset", &MLSimulator::reset)
	.def("hit_rate", &MLSimulator::hit_rate)
	.def("byte_hit_rate", &MLSimulator::byte_hit_rate)
	.def("free_space", &MLSimulator::free_space)
	.def("decide", &MLSimulator::decide)
	.def("get_ratings", &MLSimulator::get_ratings)
	.def("set_ratings", &MLSimulator::set_ratings)
	.def("get_latest_marks", &MLSimulator::get_latest_marks)
	.def("set_latest_marks", &MLSimulator::set_latest_marks)
	.def("get_updates", &MLSimulator::get_updates)
	.def("set_updates", &MLSimulator::set_updates)
	.def("get_sizes", &MLSimulator::get_sizes)
	.def("set_sizes", &MLSimulator::set_sizes)
	.def("get_used_space", &MLSimulator::get_used_space)
	.def("set_used_space", &MLSimulator::set_used_space)
	.def("get_cache_size", &MLSimulator::get_cache_size)
	.def("set_cache_size", &MLSimulator::set_cache_size)
	.def("get_L", &MLSimulator::get_L)
	.def("set_L", &MLSimulator::set_L)
	.def("get_misses", &MLSimulator::get_misses)
	.def("set_misses", &MLSimulator::set_misses)
	.def("get_hits", &MLSimulator::get_hits)
	.def("set_hits", &MLSimulator::set_hits)
	.def("get_byte_misses", &MLSimulator::get_byte_misses)
	.def("set_byte_misses", &MLSimulator::set_byte_misses)
	.def("get_byte_hits", &MLSimulator::get_byte_hits)
	.def("set_byte_hits", &MLSimulator::set_byte_hits)
	.def("get_total_rating", &MLSimulator::get_total_rating)
	.def("set_total_rating", &MLSimulator::set_total_rating)
	.def("get_ml_eviction", &MLSimulator::get_ml_eviction)
	.def("set_ml_eviction", &MLSimulator::set_ml_eviction)
	.def_readwrite("refresh_period", &MLSimulator::refresh_period)
	.def_readonly("prediction_updated_eviction", &MLSimulator::prediction_updated_eviction)
	.def_readonly("prediction_updated_admission", &MLSimulator::prediction_updated_admission);
}