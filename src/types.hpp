#ifndef PBF_TYPES_HPP
#define PBF_TYPES_HPP

#include <Eigen/Core>

using Scalar = double;
using Vec3   = Eigen::Matrix<Scalar, 3, 1>;
using VecX   = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using MatX   = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

#endif
