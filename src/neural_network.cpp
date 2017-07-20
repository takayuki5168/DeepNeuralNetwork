#include "include/neural_network.hpp"
#include "include/math_util.hpp"

namespace MachineLearning
{

void NeuralNetwork::initNetwork()
{
    in_neuron.resize(m_in_num);
    hid_neuron.resize(m_hid_num);
    out_neuron.resize(m_out_num);

    for (auto itr = in_neuron.begin(); itr != in_neuron.end(); itr++) {
        itr->initNeuron(1, MathUtil::sigmoid);
    }

    for (auto itr = hid_neuron.begin(); itr != hid_neuron.end(); itr++) {
        itr->initNeuron(m_in_num, MathUtil::sigmoid);
    }

    for (auto itr = out_neuron.begin(); itr != out_neuron.end(); itr++) {
        itr->initNeuron(m_hid_num, MathUtil::sigmoid);
    }
}

void NeuralNetwork::forward()
{
}

void NeuralNetwork::backward()
{
}

double NeuralNetwork::softmax()
{
    return 1.0;
}

}  // namespace of MachineLearning
