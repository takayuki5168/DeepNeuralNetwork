#pragma once

#include "dnn/abst_layer.hpp"

namespace MachineLearning
{

class Dense : public AbstLayer
{
public:
    explicit Dense(int neuron_num,
        ActFunc activation = ActFunc())
        : AbstLayer(neuron_num)
    {
    }
    explicit Dense(int neuron_num, int input_num,
        ActFunc activation = ActFunc())
        : AbstLayer(neuron_num, input_num) {}

    Eigen::VectorXf forward(Eigen::VectorXf in_val) override
    {
    }
    Eigen::VectorXf backward(Eigen::VectorXf in_val) override
    {
    }

private:
    ActFunc m_activation;
};

}  // namespace of MachineLearning
