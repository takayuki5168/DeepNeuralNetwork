/*!
 * @file    neuron.hpp
 * @brief   class of neuron
 */
#pragma once

#include <vector>
#include <functional>
#include <memory>
#include <Eigen/Geometry>

namespace MachineLearning
{

class Neuron
{
    //friend class NeuralNetwork;
public:
    /*!
     * @brief   initialize of neuron
     * @param   in num    num of input
     *          func      activate function
     */
    explicit Neuron(std::function<double(double)> active_func, std::vector<std::shared_ptr<Neuron>> in_neuron = {nullptr});

    double getOutValue() const { return m_out_value; }

    /*!
     * @brief   calc output of neuron
     */
    void calc();
    void calc(Eigen::VectorXd);

private:
    int m_in_num;                                 // num of input
    std::function<double(double)> m_active_func;  // activate function

    double m_bias;                // bias
    Eigen::VectorXd m_weigh_vec;  // weigh
    double m_out_value;           // output_value
    std::vector<std::shared_ptr<Neuron>> m_in_neuron_vec;
};

}  // namespace of MachineLearning
