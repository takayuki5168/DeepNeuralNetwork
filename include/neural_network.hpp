/*!
 * @file    neural_network.hpp
 * @brief   
 */

#pragma once

namespace MachineLearning
{

class NeuralNetwork
{
public:
    explicit NeuralNetwork() {}

    void initNetwork();
    void forward();
    void backward();

private:
    double softmax();
};

}  // namespace of MachineLearning
