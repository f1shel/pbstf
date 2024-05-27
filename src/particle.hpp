#ifndef PBF_PARTICLE_HPP
#define PBF_PARTICLE_HPP

#include "types.hpp"

struct Particle {
    int    i; // index
    Scalar m; // mass
    Vec3   x; // position
    Vec3   v; // velocity
    Vec3   p; // position during optim
    Vec3   n; // normal
    bool   s; // whether surface point
};

#endif
