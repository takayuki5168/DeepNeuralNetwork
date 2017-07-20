/*!
 * @file    neural_network.hpp
 * @brief   
 */

#pragma once

#include <array>
#include <vector>
#include "include/neuron.hpp"

namespace MachineLearning
{

class NeuralNetwork
{
public:
    explicit NeuralNetwork(int in_num, int hid_num, int out_num)
        : m_in_num(in_num), m_hid_num(hid_num), m_out_num(out_num) {}

    void initNetwork();
    void forward();
    void backward();

private:
    double softmax();

    // 入力層、中間層、出力層の数
    int m_in_num, m_hid_num, m_out_num;

    // 入力、中間、出力のニューロン
    std::vector<Neuron> in_neuron;
    std::vector<Neuron> hid_neuron;
    std::vector<Neuron> out_neuron;
};

}  // namespace of MachineLearning
