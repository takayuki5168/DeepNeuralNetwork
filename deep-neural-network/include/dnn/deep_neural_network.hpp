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
    void add(std::unique_ptr<AbstLayer>);

    void train();
    void test();

private:
    void forward();
    void backward();
};

}  // namespace of MachineLearning
