#pragma once

#include "dnn/abst_layer.hpp"

namespace MachineLearning
{

class Dropout : public AbstLayer
{
public:
    explicit Dropout(double rate)
        : AbstLayer(), m_rate(rate) {}

    Eigen::VectorXd forward(Eigen::VectorXd in_val) override
    {
        return in_val;
    }
    Eigen::VectorXd backward(Eigen::VectorXd in_val) override
    {
        return in_val;
    }

private:
    const double m_rate = 0.0;
};

}  // namespace of MachineLearning
