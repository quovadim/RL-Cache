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
    virtual void produce_new_cache_state(p::dict &request, double eviction_rating, bool admission_rating);
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
	.def_readwrite("refresh_period", &LRUSimulator::refresh_period)
	.def_readonly("prediction_updated_eviction", &LRUSimulator::prediction_updated_eviction)
	.def_readonly("prediction_updated_admission", &LRUSimulator::prediction_updated_admission);
}