/*!
 * @file    neural_network.hpp
 * @brief   class of neural network
 */
#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <Eigen/Geometry>
#include "include/neuron.hpp"

namespace MachineLearning
{

class NeuralNetwork
{
public:
    explicit NeuralNetwork() {}
    void addLayer(const int neuron_num, std::function<double(std::vector<double>)> active_func);

    void forward(Eigen::VectorXd in_value_vec);
    void backward();

private:
    std::vector<std::vector<std::shared_ptr<Neuron>>> m_layer_vec;
};

}  // namespace of MachineLearning
