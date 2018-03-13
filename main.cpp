#include <iostream>
#include "dnn/deep_neural_network.hpp"
#include "dnn/math_util.hpp"

int main()
{
    using namespace MachineLearning;

    std::unique_ptr<DeepNeuralNetwork> dnn = std::make_unique<DeepNeuralNetwork>();

    /*
    auto dnn = std::make_unique<NeuralNetwork>();
    dnn->add(Dense(2));
    dnn->add(LSTM(2));
    dnn->add(Activatioin("softmax"));
    */

    return 0;
}
