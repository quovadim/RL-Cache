#include "S4LRUSim.h"

SLRUSimulator::SLRUSimulator(uint64_t _cache_size) :
    SNLRUSimulator(_cache_size, 2)
{}