#pragma once

#include "dnn/abst_layer.hpp"

namespace MachineLearning
{

class Dense : public AbstLayer
{
public:
    explicit Dense(int neuron_num)
        : AbstLayer(), m_neuron_num(neuron_num) {}

    void forward() {}
    void backward() {}
private:
    const int m_neuron_num = 1;
};

}  // namespace of MachineLearning
