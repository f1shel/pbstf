#include "kernels.hpp"

Scalar calcPoly6Kernel(const Vec3& r, const Scalar h)
{
    constexpr Scalar pi    = 3.14159265358979323;
    constexpr Scalar coeff = 315.0 / (64.0 * pi);

    const Scalar h_squared = h * h;
    const Scalar r_squared = r.squaredNorm();

    if (r_squared > h_squared) {
        return 0.0;
    }

    const Scalar h_4th_power = h_squared * h_squared;
    const Scalar h_9th_power = h_4th_power * h_4th_power * h;
    const Scalar diff        = h_squared - r_squared;
    const Scalar diff_cubed  = diff * diff * diff;

    return (coeff / h_9th_power) * diff_cubed;
}

Vec3 calcGradPoly6Kernel(const Vec3& r, const Scalar h)
{
    constexpr Scalar pi    = 3.14159265358979323;
    constexpr Scalar coeff = 945.0 / (32.0 * pi);

    const Scalar h_squared = h * h;
    const Scalar r_squared = r.squaredNorm();

    if (r_squared > h_squared) {
        return Vec3::Zero();
    }

    const Scalar h_4th_power = h_squared * h_squared;
    const Scalar h_9th_power = h_4th_power * h_4th_power * h;

    const Scalar diff         = h_squared - r_squared;
    const Scalar diff_squared = diff * diff;

    return -r * (coeff / h_9th_power) * diff_squared;
}

Scalar calcSpikyKernel(const Vec3& r, const Scalar h)
{
    constexpr Scalar pi    = 3.14159265358979323;
    constexpr Scalar coeff = 15.0 / pi;

    const Scalar r_norm = r.norm();

    if (r_norm > h) {
        return 0.0;
    }

    const Scalar h_cubed     = h * h * h;
    const Scalar h_6th_power = h_cubed * h_cubed;
    const Scalar diff        = h - r_norm;
    const Scalar diff_cubed  = diff * diff * diff;

    return (coeff / h_6th_power) * diff_cubed;
}

Vec3 calcGradSpikyKernel(const Vec3& r, const Scalar h)
{
    constexpr Scalar pi    = 3.14159265358979323;
    constexpr Scalar coeff = 45.0 / pi;

    const Scalar r_norm = r.norm();

    if (r_norm > h) {
        return Vec3::Zero();
    }

    const Scalar h_cubed      = h * h * h;
    const Scalar h_6th_power  = h_cubed * h_cubed;
    const Scalar diff         = h - r_norm;
    const Scalar diff_squared = diff * diff;

    return -r * (coeff / (h_6th_power * (r_norm > 1e-24f? r_norm:1e-24f))) * diff_squared;
}

Scalar calcCubicSplineKernel(const Vec3& r, const Scalar h)
{
    constexpr Scalar pi    = 3.14159265358979323;
    const Scalar     sigma = 8.0 / (pi * h * h * h);
    const Scalar     q     = r.norm() / h;

    if (q >= 0 && q <= 0.5) {
        return sigma * (6 * q * q * (q - 1) + 1);
    } else if (q > 0.5 && q <= 1) {
        const Scalar one_minus_q = 1 - q;
        return sigma * (2 * one_minus_q * one_minus_q * one_minus_q);
    }
    return 0.0;
}

Vec3 calcGradCubicSplineKernel(const Vec3& r, const Scalar h)
{
    constexpr Scalar pi     = 3.14159265358979323;
    const Scalar     sigma  = 8.0 / (pi * h * h * h);
    const Scalar     r_norm = r.norm();
    const Scalar     q      = r_norm / h;
    const Vec3       dpdq   = r / (h * (r_norm > 1e-24f ? r_norm : 1e-24f));
    if (q > 1) {
        return Vec3::Zero();
    } else if (q <= 0.5) {
        return sigma * (6 * q * (3 * q - 2)) * dpdq;
    } else if (q > 0.5 && q <= 1) {
        const Scalar one_minus_q = 1 - q;
        return sigma * (6 * one_minus_q * one_minus_q * (-1)) * dpdq;
    }
}