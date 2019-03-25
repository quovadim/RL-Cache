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

class S4LRUSimulator : public SNLRUSimulator {
public:
	S4LRUSimulator(uint64_t _cache_size);
};

BOOST_PYTHON_MODULE(S4LRUSim) {
	using namespace boost::python;

	class_<S4LRUSimulator>("S4LRUSimulator", init<uint64_t>())
	.def("__copy__", &generic__copy__< S4LRUSimulator >)
    .def("__deepcopy__", &generic__deepcopy__< S4LRUSimulator >)
	.def("reset", &S4LRUSimulator::reset)
	.def("hit_rate", &S4LRUSimulator::hit_rate)
	.def("byte_hit_rate", &S4LRUSimulator::byte_hit_rate)
	.def("free_space", &S4LRUSimulator::free_space)
	.def("decide", &S4LRUSimulator::decide)
    .def("get_ratings", &S4LRUSimulator::get_ratings)
	.def("set_ratings", &S4LRUSimulator::set_ratings)
	.def("get_sizes", &S4LRUSimulator::get_sizes)
	.def("set_sizes", &S4LRUSimulator::set_sizes)
	.def("get_used_space", &S4LRUSimulator::get_used_space)
	.def("set_used_space", &S4LRUSimulator::set_used_space)
	.def("get_cache_size", &S4LRUSimulator::get_cache_size)
	.def("set_cache_size", &S4LRUSimulator::set_cache_size)
	.def("get_L", &S4LRUSimulator::get_L)
	.def("set_L", &S4LRUSimulator::set_L)
	.def("get_misses", &S4LRUSimulator::get_misses)
	.def("set_misses", &S4LRUSimulator::set_misses)
	.def("get_hits", &S4LRUSimulator::get_hits)
	.def("set_hits", &S4LRUSimulator::set_hits)
	.def("get_byte_misses", &S4LRUSimulator::get_byte_misses)
	.def("set_byte_misses", &S4LRUSimulator::set_byte_misses)
	.def("get_byte_hits", &S4LRUSimulator::get_byte_hits)
	.def("set_byte_hits", &S4LRUSimulator::set_byte_hits)
	.def("get_total_rating", &S4LRUSimulator::get_total_rating)
	.def("set_total_rating", &S4LRUSimulator::set_total_rating)
	.def("get_ml_eviction", &S4LRUSimulator::get_ml_eviction)
	.def("set_ml_eviction", &S4LRUSimulator::set_ml_eviction)
	.def_readwrite("refresh_period", &S4LRUSimulator::refresh_period)
	.def_readonly("prediction_updated_eviction", &S4LRUSimulator::prediction_updated_eviction)
	.def_readonly("prediction_updated_admission", &S4LRUSimulator::prediction_updated_admission);
}