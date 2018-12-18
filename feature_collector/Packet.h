//
// Created by vadim on 6/04/18.
//

#pragma once
#include <stdint.h>

struct Packet{
    Packet(uint64_t _id, uint64_t _size, uint64_t _timestamp) :
            id(_id),
            size(_size),
            timestamp(_timestamp)
    {}

    uint64_t id;
    uint64_t size;
    uint64_t timestamp;
};
