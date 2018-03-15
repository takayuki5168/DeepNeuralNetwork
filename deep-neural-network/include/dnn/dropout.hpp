#pragma once

#include "dnn/abst_layer.hpp"

namespace MachineLearning
{

class Dropout : public AbstLayer
{
public:
    explicit Dropout(double rate)
        : AbstLayer(), m_rate(rate) {}

    Eigen::VectorXf forward(Eigen::VectorXf in_val) override
    {
        in_val[0];
    }
    Eigen::VectorXf backward(Eigen::VectorXf in_val) override
    {
    }

private:
    const double m_rate = 0.0;
};

}  // namespace of MachineLearning
