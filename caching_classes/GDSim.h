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


class GDSimulator : public CacheSim {
public:

	GDSimulator(uint64_t _cache_size);

protected:

    virtual void produce_new_cache_state(p::dict &request, double eviction_rating, bool admission_decision);

};

BOOST_PYTHON_MODULE(GDSim) {
	using namespace boost::python;

	class_<GDSimulator>("GDSimulator", init<uint64_t>())
	.def("__copy__", &generic__copy__< GDSimulator >)
    .def("__deepcopy__", &generic__deepcopy__< GDSimulator >)
	.def("reset", &GDSimulator::reset)
	.def("hit_rate", &GDSimulator::hit_rate)
	.def("byte_hit_rate", &GDSimulator::byte_hit_rate)
	.def("free_space", &GDSimulator::free_space)
	.def("decide", &GDSimulator::decide)
	.def("get_ratings", &GDSimulator::get_ratings)
	.def("set_ratings", &GDSimulator::set_ratings)
	.def_readwrite("refresh_period", &GDSimulator::refresh_period)
	.def_readonly("prediction_updated_eviction", &GDSimulator::prediction_updated_eviction)
	.def_readonly("prediction_updated_admission", &GDSimulator::prediction_updated_admission);
}