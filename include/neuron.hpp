/*!
 * @file    neuron.hpp
 * @brief   
 */
#pragma once

#include <vector>
#include <functional>
#include <Eigen/Geometry>

namespace MachineLearning
{

class Neuron
{
public:
    //explicit Neuron(int in_num, std::function<double(double)> active_func)
    //    : m_in_num(in_num), m_active_func(active_func) {}
    //explicit Neuron(int in_num)
    //    : m_in_num(in_num) {}

    void initNeuron(int in_num, std::function<double(double)> func);
    //void initNeuron(std::function<double(double)> func);

private:
    double calc();

    int m_in_num;
    std::function<double(double)> m_active_func;

    double m_bias;
    Eigen::VectorXd m_in_value;
    Eigen::VectorXd m_weigh;
};

}  // namespace of MachineLearning
