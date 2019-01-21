#include "AdaptSizeSim.h"

#include <unordered_map>
#include <iostream>
#include <math.h>

using std::unordered_map;
using std::pow;
using std::log2;
using std::fabs;
using std::exp;

using std::cerr;
using std::endl;

#define SHFT2(a,b,c) (a)=(b);(b)=(c);
#define SHFT3(a,b,c,d) (a)=(b);(b)=(c);(c)=(d);

static inline double oP1(double T, double l, double p) {
  return (l * p * T * (840.0 + 60.0 * l * T + 20.0 * l*l * T*T + l*l*l * T*T*T));
}

static inline double oP2(double T, double l, double p) {
  return (840.0 + 120.0 * l * (-3.0 + 7.0 * p) * T + 60.0 * l*l * (1.0 + p) * T*T + 4.0 * l*l*l * (-1.0 + 5.0 * p) * T*T*T + l*l*l*l * p * T*T*T*T);
}

AdaptSizeSimulator::AdaptSizeSimulator(uint64_t _cache_size) :
    CacheSim(_cache_size),
    stat_size(0),
    next_configuration(configuration_interval),
    c(1 << 15)
{}

bool AdaptSizeSimulator::decide(p::dict request, double eviction_rating, int admission_decision) {

    reconfigure();

    uint64_t id = p::extract<uint64_t>(request.get("id"));
    uint64_t size = p::extract<uint64_t>(request.get("size"));

    if (emwa_info.count(id) == 0 && intervals_info.count(id) == 0) {
        stat_size += size;
    }

    auto& info = intervals_info[id];
    info.count += 1;
    info.size = size;

    return CacheSim::decide(request, eviction_rating, admission_decision);
}

void AdaptSizeSimulator::reconfigure() {
    next_configuration--;
    if (next_configuration) {
        return;
    } else if (stat_size < 3 * cache_size) {
        next_configuration += 10000;
        return;
    }

    next_configuration = configuration_interval;
    for (auto it = emwa_info.begin(); it != emwa_info.end(); it++) {
        it->second.count *= decay;
    }

    for (auto it = intervals_info.begin(); it != intervals_info.end(); it++) {
        auto emwa_it = emwa_info.find(it->first);
        if (emwa_it != emwa_info.end()) {
            emwa_it->second.count += (1. - decay) * it->second.count;
            emwa_it->second.size = it->second.size;
        } else {
            emwa_info.insert(*it);
        }
    }

    intervals_info.clear();

    aligned_count.clear();
    aligned_size.clear();

    double total_count = 0;
    uint64_t total_size = 0;

    for (auto it = emwa_info.begin(); it != emwa_info.end();) {
        if (it->second.count < 0.1) {
            stat_size -= it->second.size;
            it = emwa_info.erase(it);
        } else {
            aligned_count.push_back(it->second.count);
            total_count += it->second.count;
            aligned_size.push_back(it->second.size);
            total_size += it->second.size;
            it++;
        }
    }

    double v = 1. - r;

    double x0 = 0;
	double x1 = log2(cache_size);
	double x2 = x1;
	double x3 = x1;

	double bestHitRate = 0.0;

	for (int i = 2; i < x3; i += 4) {
		const double next_log2c = i;
		const double hitRate = model_hit_rate(next_log2c);

		if(hitRate > bestHitRate) {
			bestHitRate = hitRate;
			x1 = next_log2c;
		}
	}

	double h1 = bestHitRate;
	double h2;

	if(x3-x1 > x1-x0) {

		x2 = x1 + v * (x3 - x1);
		h2 = model_hit_rate(x2);
	} else {

		x2 = x1;
		h2 = h1;
		x1 = x0 + v * (x1 - x0);
		h1 = model_hit_rate(x1);
	}

	int curIterations=0;
	// use termination condition from [Numerical recipes in C]
	while (curIterations++ < max_iterations && fabs(x3 - x0) > tol * (fabs(x1) + fabs(x2))) {
		//NAN check
		if ((h1 != h1) || (h2 != h2))
			break;

		if(h2 > h1) {
			SHFT3(x0, x1, x2, r * x1 + v * x3);
			SHFT2(h1, h2, model_hit_rate(x2));
		} else {
			SHFT3(x3, x2, x1, r * x2 + v * x0);
			SHFT2(h2, h1, model_hit_rate(x1));
		}
	}

	// check result
	if( (h1 != h1) || (h2 != h2) ) {
		;
		// nop
	} else if (h1 > h2) {
		// x1 should is final parameter
		c = pow(2, x1);
	} else {
		c = pow(2, x2);
	}
}

double AdaptSizeSimulator::model_hit_rate(double log2c) {
    // this code is adapted from the AdaptSize git repo
    // github.com/dasebe/AdaptSize
    double old_T, the_T, the_C;
    double sum_val = 0.;
    double thparam = log2c;

    for (size_t i = 0; i < aligned_count.size(); i++) {
        sum_val += aligned_count[i] * (exp(-aligned_size[i] / pow(2, thparam))) * aligned_size[i];
    }

    if (sum_val <= 0) {
        return(0);
    }
    the_T = double(cache_size) / sum_val;
    // prepare admission probabilities
    aligned_ohr.clear();
    for (size_t i = 0; i < aligned_count.size(); i++) {
        aligned_ohr.push_back(exp(-aligned_size[i] / pow(2.0, thparam)));
    }
    // 20 iterations to calculate TTL

    for (int j = 0; j < 10; j++) {
        the_C = 0;
        if(the_T > 1e70) {
            break;
        }
        for (size_t i=0; i < aligned_count.size(); i++) {
            const double reqTProd = aligned_count[i] * the_T;
            if (reqTProd > 150) {
                // cache hit probability = 1, but numerically inaccurate to calculate
                the_C += aligned_size[i];
            } else {
                const double expTerm = exp(reqTProd) - 1;
                const double expAdmProd = aligned_ohr[i] * expTerm;
                const double tmp = expAdmProd / (1 + expAdmProd);
                the_C += aligned_size[i] * tmp;
            }
        }
        old_T = the_T;
        the_T = double(cache_size) * old_T / the_C;
    }

    // calculate object hit ratio
    double weighted_hitratio_sum = 0;
    for (size_t i=0; i < aligned_count.size(); i++) {
        const double tmp01 = oP1(the_T, aligned_count[i], aligned_ohr[i]);
        const double tmp02 = oP2(the_T, aligned_count[i], aligned_ohr[i]);
        double tmp = 0;
        if (tmp01 !=0 && tmp02 == 0)
            tmp = 0.0;
        else
            tmp = tmp01 / tmp02;

        if (tmp < 0.0)
            tmp = 0.0;
        else if (tmp > 1.0)
            tmp = 1.0;

        weighted_hitratio_sum += aligned_count[i] * tmp;
    }
  return weighted_hitratio_sum;
}

bool AdaptSizeSimulator::admit(p::dict &request) {
    double roll = distr(generator);
    uint64_t size = p::extract<uint64_t>(request.get("size"));
	double admitProb = exp(-1.0 * double(size) / c);

	return roll < admitProb;
}

void AdaptSizeSimulator::produce_new_cache_state(p::dict &request, double eviction_rating, int admission_decision) {
	uint64_t size = p::extract<uint64_t>(request.get("size"));

	if (!admit(request)) {
        	return;
	}

	prediction_updated_eviction = true;

	while (used_space + size > cache_size) {
	    auto min_elem = ratings.begin();

	    used_space -= sizes[min_elem->second];

		cache.erase(min_elem->second);
		sizes.erase(min_elem->second);
		ratings.erase(min_elem);
	}

	double prediction = eviction_rating;
	uint64_t id = p::extract<uint64_t>(request.get("id"));

    double rating = prediction;
    total_rating += rating;
    cache[id] = ratings.emplace(rating, id);
	sizes.insert(pair<uint64_t, uint64_t>(id, size));
	used_space += size;

	return;
}
