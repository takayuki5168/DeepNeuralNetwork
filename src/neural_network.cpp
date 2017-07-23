#include "include/neural_network.hpp"
#include "include/math_util.hpp"

namespace MachineLearning
{

void NeuralNetwork::initNetwork()
{
    // resize input, hidden, output layer
    m_in_neuron.resize(m_in_num);
    m_hid_neuron.resize(m_hid_num);
    m_out_neuron.resize(m_out_num);

    // init neuron of input layer
    for (auto itr = m_in_neuron.begin(); itr != m_in_neuron.end(); itr++) {
        itr->initNeuron(1, MathUtil::identity);
    }

    // init neuron of hidden layer
    for (auto itr = m_hid_neuron.begin(); itr != m_hid_neuron.end(); itr++) {
        itr->initNeuron(m_in_num, MathUtil::sigmoid);
    }

    // init neuron of output layer
    for (auto itr = m_out_neuron.begin(); itr != m_out_neuron.end(); itr++) {
        itr->initNeuron(m_hid_num, MathUtil::sigmoid);
    }
}

void NeuralNetwork::forward(Eigen::VectorXd in_in_value)
{
    Eigen::VectorXd hid_in_value;
    Eigen::VectorXd out_in_value;

    // input value of neuron of input layer
    in_in_value.resize(m_in_num);
    // input value of neuron of hidden layer
    hid_in_value.resize(m_in_num);
    // input value of neuron of output layer
    out_in_value.resize(m_hid_num);

    // calc of neuron of hidden layer
    for (int i = 0; i < m_in_num; i++) {
        m_in_neuron.at(i).m_in_value(0) = in_in_value(i);
        m_in_neuron.at(i).calc();
        for (int j = 0; j < m_hid_num; j++) {
            m_hid_neuron.at(i).m_in_value(j) = m_in_neuron.at(i).m_out_value;
        }
    }

    // calc of neuron of output layer
    for (int i = 0; i < m_hid_num; i++) {
        for (int j = 0; j > m_in_num; j++) {
            m_hid_neuron.at(i).m_in_value(j) = hid_in_value(j);
        }
        m_hid_neuron.at(i).calc();
        for (int k = 0; k < m_out_num; k++) {
            m_out_neuron.at(i).m_in_value(k) = m_hid_neuron.at(i).m_out_value;
        }
    }
}

void NeuralNetwork::backward()
{
}

Eigen::VectorXd NeuralNetwork::softmax()
{
    // calc max value of output
    double max_output = 0;
    for (int i = 0; i < m_out_num; i++) {
        max_output = std::max(max_output, m_out_neuron.at(i).m_out_value);
    }

    double sum = 0;
    for (int i = 0; i < m_out_num; i++) {
        sum += std::exp(m_out_neuron.at(i).m_out_value - max_output);
    }

    Eigen::VectorXd output;
    output.resize(m_out_num);
    for (int i = 0; i < m_out_num; i++) {
        output(i) = std::exp(m_out_neuron.at(i).m_out_value - max_output) / sum;
    }

    return output;
}

}  // namespace of MachineLearning
