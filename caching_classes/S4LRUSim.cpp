#include "S4LRUSim.h"

S4LRUSimulator::S4LRUSimulator(uint64_t _cache_size) :
    SNLRUSimulator(_cache_size, 4)
{}