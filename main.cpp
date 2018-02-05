#include <iostream>
#include "include/neural_network.hpp"
#include "include/math_util.hpp"

int main()
{
    auto neural_network = std::make_unique<MachineLearning::NeuralNetwork>();
    neural_network->addLayer(24 * 24, MathUtil::identity);
    neural_network->addLayer(100, MathUtil::relu);
    neural_network->addLayer(10, MathUtil::relu);
    neural_network->addLayer(10, MathUtil::softmax);

    return 0;
}
