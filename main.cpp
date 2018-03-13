#include <iostream>
#include "dnn/deep_neural_network.hpp"
#include "dnn/dense.hpp"
#include "dnn/dropout.hpp"

int main()
{
    using namespace MachineLearning;

    std::unique_ptr<DeepNeuralNetwork> dnn = std::make_unique<DeepNeuralNetwork>();
    dnn->add(std::make_unique<AbstLayer>());
    dnn->add(std::make_unique<Dense>(10));
    dnn->add(std::make_unique<Dropout>(0.1));

    //dnn->compile(Crossentropy(), Adam());

    //dnn->fit();
    //dnn->predict();

    return 0;
}
