#pragma once

#include "dnn/abst_layer.hpp"

namespace MachineLearning
{

class Dense : public AbstLayer
{
public:
    explicit Dense(int neuron_num)
        : AbstLayer(neuron_num) {}
    explicit Dense(int neuron_num, int input_num)
        : AbstLayer(neuron_num, input_num) {}

    Eigen::VectorXd forward(Eigen::VectorXd in_mat) override
    {
        Eigen::VectorXd out_mat;
        out_mat = in_mat * m_weights;
        return out_mat;
    }
    // TODO
    Eigen::VectorXd backward(Eigen::VectorXd in_mat) override
    {
        return in_mat;
    }

private:
};

}  // namespace of MachineLearning
