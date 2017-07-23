/*!
 * @file    neuron.hpp
 * @brief   class of neuron
 */
#pragma once

#include <vector>
#include <functional>
#include <Eigen/Geometry>

namespace MachineLearning
{

class Neuron
{
    friend class NeuralNetwork;

private:
    /*!
     * @brief   initialize of neuron
     * @param   in num    num of input
     *          func      activate function
     */
    void initNeuron(int in_num, std::function<double(double)> func);
    /*!
     * @brief   calc output of neuron
     */
    void calc();

    int m_in_num;                                 // num of input
    std::function<double(double)> m_active_func;  // activate function

    double m_bias;               // bias
    Eigen::VectorXd m_weigh;     // weigh
    Eigen::VectorXd m_in_value;  // input value
    double m_out_value;          // output_value
};

}  // namespace of MachineLearning
