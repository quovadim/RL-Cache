//
// Created by vadim on 6/04/18.
//

#include "InfoCollector.h"

#include <algorithm>
#include <stdint.h>
#include <math.h>

using std::find;
using std::min;
using std::max;
using std::pair;
using std::log2;

void InfoCollector::update_info(Packet* packet) {
    auto it = frequency.find(packet->id);
    if (it == frequency.end()) {
        frequency.insert(pair<uint64_t, uint64_t>(packet->id, 1));
        unique_bytes += packet->size;
        unique_requests++;
    } else {
        double n = it->second;
        entropy_sum -= n * log2(n);
        entropy_sum += (n + 1) * log2(n + 1);
        it->second++;
    }

    requests++;
    bytes += packet->size;
}
