#pragma once

#include "dnn/abst_layer.hpp"

namespace MachineLearning
{

class Dense : public AbstLayer
{
public:
    explicit Dense(int neuron_num)
        : AbstLayer(neuron_num)
    {
    }
    explicit Dense(int neuron_num, int input_num)
        : AbstLayer(neuron_num, input_num) {}

    Eigen::VectorXd forward(Eigen::VectorXd in_val) override
    {
        return in_val;
    }
    Eigen::VectorXd backward(Eigen::VectorXd in_val) override
    {
        return in_val;
    }

private:
};

}  // namespace of MachineLearning
