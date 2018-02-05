#include "include/neural_network.hpp"

namespace MachineLearning
{

void NeuralNetwork::addLayer(const int neuron_num, std::function<double(double)> active_func)
{
    std::vector<std::shared_ptr<Neuron>> neuron_vec;
    neuron_vec.resize(neuron_num);
    for (auto neuron : neuron_vec) {
        if (neuron_vec.size() == 0) {
            neuron = std::make_shared<Neuron>(active_func, m_layer_vec.back());
        } else {
            neuron = std::make_shared<Neuron>(active_func);
        }
    }
    m_layer_vec.push_back(neuron_vec);
}

void NeuralNetwork::forward(Eigen::VectorXd in_value_vec)
{
    for (unsigned int i = 0; i < m_layer_vec.size(); i++) {
        for (unsigned int j = 0; j < m_layer_vec.at(i).size(); j++) {
            if (i == 0) {
                m_layer_vec.at(i).at(j)->calc(in_value_vec);
            } else {
                m_layer_vec.at(i).at(j)->calc();
            }
        }
    }
}

void NeuralNetwork::backward()
{
}

/*
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
*/
}  // namespace of MachineLearning
