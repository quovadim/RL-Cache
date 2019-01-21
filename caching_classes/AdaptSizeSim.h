#include <Python.h>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <map>
#include <unordered_map>
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
using std::unordered_map;

namespace p = boost::python;

typedef map<uint64_t, double> dict_predictions;
typedef map<uint64_t, uint64_t> dict_sizes;

struct object_info {

    object_info() :
        size(0),
        count(0)
    {}

    uint64_t size;
    double count;
};

typedef unordered_map<uint64_t, object_info> ads_info;
typedef vector<double> aligned_info;

class AdaptSizeSimulator : public CacheSim{
public:

	AdaptSizeSimulator(uint64_t _cache_size);

	virtual bool decide(p::dict request, double eviction_rating, int admission_decision);

protected:
    virtual void produce_new_cache_state(p::dict &request, double eviction_rating, int admission_rating);

    void reconfigure();

    double model_hit_rate(double c);

    bool admit(p::dict &request);

    const uint64_t configuration_interval = 500000;
    const double decay = 0.3;
    const double r = 0.61803399;
    const double tol = 3.0e-8;
    const int max_iterations = 15;

    uint64_t stat_size;
    uint64_t next_configuration;
    double c;

    ads_info emwa_info;
    ads_info intervals_info;

    aligned_info aligned_count;
    aligned_info aligned_size;
    aligned_info aligned_ohr;
};

BOOST_PYTHON_MODULE(AdaptSizeSim) {
	using namespace boost::python;

	class_<AdaptSizeSimulator>("AdaptSizeSimulator", init<uint64_t>())
	.def("__copy__", &generic__copy__< AdaptSizeSimulator >)
    .def("__deepcopy__", &generic__deepcopy__< AdaptSizeSimulator >)
	.def("reset", &AdaptSizeSimulator::reset)
	.def("hit_rate", &AdaptSizeSimulator::hit_rate)
	.def("byte_hit_rate", &AdaptSizeSimulator::byte_hit_rate)
	.def("free_space", &AdaptSizeSimulator::free_space)
	.def("decide", &AdaptSizeSimulator::decide)
    .def("get_ratings", &AdaptSizeSimulator::get_ratings)
	.def("set_ratings", &AdaptSizeSimulator::set_ratings)
	.def("get_sizes", &AdaptSizeSimulator::get_sizes)
	.def("set_sizes", &AdaptSizeSimulator::set_sizes)
	.def("get_used_space", &AdaptSizeSimulator::get_used_space)
	.def("set_used_space", &AdaptSizeSimulator::set_used_space)
	.def("get_cache_size", &AdaptSizeSimulator::get_cache_size)
	.def("set_cache_size", &AdaptSizeSimulator::set_cache_size)
	.def("get_L", &AdaptSizeSimulator::get_L)
	.def("set_L", &AdaptSizeSimulator::set_L)
	.def("get_misses", &AdaptSizeSimulator::get_misses)
	.def("set_misses", &AdaptSizeSimulator::set_misses)
	.def("get_hits", &AdaptSizeSimulator::get_hits)
	.def("set_hits", &AdaptSizeSimulator::set_hits)
	.def("get_byte_misses", &AdaptSizeSimulator::get_byte_misses)
	.def("set_byte_misses", &AdaptSizeSimulator::set_byte_misses)
	.def("get_byte_hits", &AdaptSizeSimulator::get_byte_hits)
	.def("set_byte_hits", &AdaptSizeSimulator::set_byte_hits)
	.def("get_total_rating", &AdaptSizeSimulator::get_total_rating)
	.def("set_total_rating", &AdaptSizeSimulator::set_total_rating)
	.def("get_ml_eviction", &AdaptSizeSimulator::get_ml_eviction)
	.def("set_ml_eviction", &AdaptSizeSimulator::set_ml_eviction)
	.def_readwrite("refresh_period", &AdaptSizeSimulator::refresh_period)
	.def_readonly("prediction_updated_eviction", &AdaptSizeSimulator::prediction_updated_eviction)
	.def_readonly("prediction_updated_admission", &AdaptSizeSimulator::prediction_updated_admission);
}