/*!
 * @file    deep_neural_network.hpp
 * @brief   class of DeepNeuralNetwork
 */
#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <Eigen/Geometry>
#include "dnn/abst_layer.hpp"

namespace MachineLearning
{

class DeepNeuralNetwork
{
public:
    explicit DeepNeuralNetwork() {}
    void add(std::unique_ptr<AbstLayer>&& layer) { m_layers.push_back(std::move(layer)); }

    void fit() {}
    void predict() {}

    //void compile(std::function<> loss, std::unique_ptr<Optimizer> optimizer) {}

private:
    std::vector<std::unique_ptr<AbstLayer>> m_layers;
};

}  // namespace of MachineLearning
