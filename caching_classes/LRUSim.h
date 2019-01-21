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
    virtual void produce_new_cache_state(p::dict &request, double eviction_rating, int admission_rating);
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
    .def("get_ratings", &LRUSimulator::get_ratings)
	.def("set_ratings", &LRUSimulator::set_ratings)
	.def("get_sizes", &LRUSimulator::get_sizes)
	.def("set_sizes", &LRUSimulator::set_sizes)
	.def("get_used_space", &LRUSimulator::get_used_space)
	.def("set_used_space", &LRUSimulator::set_used_space)
	.def("get_cache_size", &LRUSimulator::get_cache_size)
	.def("set_cache_size", &LRUSimulator::set_cache_size)
	.def("get_L", &LRUSimulator::get_L)
	.def("set_L", &LRUSimulator::set_L)
	.def("get_misses", &LRUSimulator::get_misses)
	.def("set_misses", &LRUSimulator::set_misses)
	.def("get_hits", &LRUSimulator::get_hits)
	.def("set_hits", &LRUSimulator::set_hits)
	.def("get_byte_misses", &LRUSimulator::get_byte_misses)
	.def("set_byte_misses", &LRUSimulator::set_byte_misses)
	.def("get_byte_hits", &LRUSimulator::get_byte_hits)
	.def("set_byte_hits", &LRUSimulator::set_byte_hits)
	.def("get_total_rating", &LRUSimulator::get_total_rating)
	.def("set_total_rating", &LRUSimulator::set_total_rating)
	.def("get_ml_eviction", &LRUSimulator::get_ml_eviction)
	.def("set_ml_eviction", &LRUSimulator::set_ml_eviction)
	.def_readwrite("refresh_period", &LRUSimulator::refresh_period)
	.def_readonly("prediction_updated_eviction", &LRUSimulator::prediction_updated_eviction)
	.def_readonly("prediction_updated_admission", &LRUSimulator::prediction_updated_admission);
}