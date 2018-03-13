#pragma once

#include <cmath>
#include <Eigen/Geometry>

namespace MathUtil
{

double identity(double x) { return x; }
double step(double x) { return (x > 0) ? 1 : 0; }
double sigmoid(double x) { return 1.0 / (1 + std::exp(-x)); }
double relu(double x) { return (x > 0) ? x : 0; }
double tanh(double x) { return std::tanh(x); }

}  // namespace of MathUtil
