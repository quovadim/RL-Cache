//
// Created by vadim on 6/04/18.
//

#pragma once
#include <stdint.h>

struct Packet{
    Packet(uint64_t _id, uint64_t _size, uint64_t _timestamp,
    uint64_t _l1, uint64_t _l2, uint64_t _l3, double _l4, double _l5, double _l6) :
            id(_id),
            size(_size),
            timestamp(_timestamp),
            l1(_l1),
            l2(_l2),
            l3(_l3),
            l4(_l4),
            l5(_l5),
            l6(_l6),
            rating(0)
    {}

    uint64_t id;
    uint64_t size;
    uint64_t timestamp;
    uint64_t l1;
    uint64_t l2;
    uint64_t l3;
    double l4;
    double l5;
    double l6;

    double rating;
};
