/*!
 * @file    neural_network.hpp
 * @brief   class of neural network
 */

#pragma once

#include <array>
#include <vector>
#include <Eigen/Geometry>
#include "include/neuron.hpp"

namespace MachineLearning
{

class NeuralNetwork
{
public:
    explicit NeuralNetwork(int in_num, int hid_num, int out_num)
        : m_in_num(in_num), m_hid_num(hid_num), m_out_num(out_num) {}

    void initNetwork();
    void forward(Eigen::VectorXd in_in_value);
    void backward();

private:
    Eigen::VectorXd softmax();

    // num of neuron of input, hidden, output layer
    int m_in_num, m_hid_num, m_out_num;

    // neuron of input, hidden, output layer
    std::vector<Neuron> m_in_neuron;
    std::vector<Neuron> m_hid_neuron;
    std::vector<Neuron> m_out_neuron;
};

}  // namespace of MachineLearning
