/*!
 * @file    math_util.hpp
 * @brief   便利な数学関数
 */

#pragma once

#include <cmath>
#include <Eigen/Geometry>

namespace MathUtil
{
//Eigen::Matrix<double, 2, 1> po;

double step(double x) { return (x > 0) ? 1 : 0; }
double sigmoid(double x) { return 1.0 / (1 + std::exp(-x)); }
double relu(double x) { return (x > 0) ? x : 0; }

}  // namespace of MathUtil
