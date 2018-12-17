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
	.def_readwrite("refresh_period", &MLSimulator::refresh_period)
	.def_readonly("prediction_updated_eviction", &MLSimulator::prediction_updated_eviction)
	.def_readonly("prediction_updated_admission", &MLSimulator::prediction_updated_admission);
}