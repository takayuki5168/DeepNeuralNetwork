#pragma once

#include "dnn/abst_layer.hpp"

namespace MachineLearning
{

class Dropout : public AbstLayer
{
public:
    explicit Dropout(double rate)
        : AbstLayer(), m_rate(rate) {}

    Eigen::VectorXd forward(Eigen::VectorXd in_mat) override
    {
        return in_mat;
    }
    Eigen::VectorXd backward(Eigen::VectorXd in_mat) override
    {
        return in_mat;
    }

private:
    const double m_rate = 0.0;
};

}  // namespace of MachineLearning
