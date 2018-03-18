#pragma once

#include "dnn/abst_layer.hpp"

namespace MachineLearning
{

class Dense : public AbstLayer
{
public:
    explicit Dense(int neuron_num)
        : AbstLayer(neuron_num) { m_bias_vec.resize(neuron_num); }
    explicit Dense(int neuron_num, int input_num)
        : AbstLayer(neuron_num, input_num) { m_bias_vec.resize(neuron_num); }

    Eigen::VectorXd forward(Eigen::VectorXd in_vec) override
    {
        m_out_vec = in_vec * m_weight_mat + m_bias_vec;
        return m_out_vec;
    }
    // TODO
    Eigen::VectorXd backward(Eigen::VectorXd in_vec) override
    {
        return in_vec;
    }

private:
    Eigen::VectorXd m_bias_vec;
};

}  // namespace of MachineLearning
