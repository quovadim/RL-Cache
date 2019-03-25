#pragma once

#include <Python.h>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <map>
#include <vector>
#include <math.h>
#include <random>
#include <algorithm>

#include "auxiliary.hpp"
#include "SNLRUSim.h"

using std::map;
using std::vector;
using std::exp;
using std::pair;

namespace p = boost::python;

typedef map<uint64_t, double> dict_predictions;
typedef map<uint64_t, uint64_t> dict_sizes;

class SLRUSimulator : public SNLRUSimulator{
public:
	SLRUSimulator(uint64_t _cache_size);
};

BOOST_PYTHON_MODULE(SLRUSim) {
	using namespace boost::python;

	class_<SLRUSimulator>("SLRUSimulator", init<uint64_t>())
	.def("__copy__", &generic__copy__< SLRUSimulator >)
    .def("__deepcopy__", &generic__deepcopy__< SLRUSimulator >)
	.def("reset", &SLRUSimulator::reset)
	.def("hit_rate", &SLRUSimulator::hit_rate)
	.def("byte_hit_rate", &SLRUSimulator::byte_hit_rate)
	.def("free_space", &SLRUSimulator::free_space)
	.def("decide", &SLRUSimulator::decide)
    .def("get_ratings", &SLRUSimulator::get_ratings)
	.def("set_ratings", &SLRUSimulator::set_ratings)
	.def("get_sizes", &SLRUSimulator::get_sizes)
	.def("set_sizes", &SLRUSimulator::set_sizes)
	.def("get_used_space", &SLRUSimulator::get_used_space)
	.def("set_used_space", &SLRUSimulator::set_used_space)
	.def("get_cache_size", &SLRUSimulator::get_cache_size)
	.def("set_cache_size", &SLRUSimulator::set_cache_size)
	.def("get_L", &SLRUSimulator::get_L)
	.def("set_L", &SLRUSimulator::set_L)
	.def("get_misses", &SLRUSimulator::get_misses)
	.def("set_misses", &SLRUSimulator::set_misses)
	.def("get_hits", &SLRUSimulator::get_hits)
	.def("set_hits", &SLRUSimulator::set_hits)
	.def("get_byte_misses", &SLRUSimulator::get_byte_misses)
	.def("set_byte_misses", &SLRUSimulator::set_byte_misses)
	.def("get_byte_hits", &SLRUSimulator::get_byte_hits)
	.def("set_byte_hits", &SLRUSimulator::set_byte_hits)
	.def("get_total_rating", &SLRUSimulator::get_total_rating)
	.def("set_total_rating", &SLRUSimulator::set_total_rating)
	.def("get_ml_eviction", &SLRUSimulator::get_ml_eviction)
	.def("set_ml_eviction", &SLRUSimulator::set_ml_eviction)
	.def_readwrite("refresh_period", &SLRUSimulator::refresh_period)
	.def_readonly("prediction_updated_eviction", &SLRUSimulator::prediction_updated_eviction)
	.def_readonly("prediction_updated_admission", &SLRUSimulator::prediction_updated_admission);
}