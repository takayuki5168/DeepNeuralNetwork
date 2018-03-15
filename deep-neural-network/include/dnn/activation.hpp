#pragma once

#include "dnn/abst_layer.hpp"
#include "dnn/math_util.hpp"

namespace MachineLearning
{

Eigen::VectorXf identify(Eigen::VectorXf in_val);
Eigen::VectorXf step(Eigen::VectorXf in_val);
Eigen::VectorXf sigmoid(Eigen::VectorXf in_val);
Eigen::VectorXf relu(Eigen::VectorXf in_val);
Eigen::VectorXf softmax(Eigen::VectorXf in_val);

class Activation : public AbstLayer
{
public:
    explicit Activation(std::string name)
        : AbstLayer() {}

    Eigen::VectorXf forward(Eigen::VectorXf in_val) override
    {
    }
    Eigen::VectorXf backward(Eigen::VectorXf in_val) override
    {
    }

private:
    const double m_rate = 0.0;
};

}  // namespace of MachineLearning
