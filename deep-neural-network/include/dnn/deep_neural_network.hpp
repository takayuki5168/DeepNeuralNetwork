/*!
 * @file    deep_neural_network.hpp
 * @brief   class of DeepNeuralNetwork
 */
#pragma once

#include <memory>
#include <functional>
#include "dnn/abst_layer.hpp"

namespace MachineLearning
{

class DeepNeuralNetwork
{
public:
    explicit DeepNeuralNetwork() {}
    void add(std::unique_ptr<AbstLayer>&& layer)
    {
        if (not m_layers.empty()) {  // This layer is not input layer
            layer->setInputNum(m_layers.back()->getInputNum());
        }
        if (layer->getNeuronNum() == 0) {  // NeuronNum was not settled
            layer->setInputNum(m_layers.back()->getNeuronNum());
            layer->setNeuronNum(m_layers.back()->getNeuronNum());
        }
        m_layers.push_back(std::move(layer));
    }

    void fit(Eigen::MatrixXd /* input_mat */)
    {
    }

    Eigen::MatrixXd predict(Eigen::MatrixXd input_mat)
    {
        Eigen::MatrixXd output_mat;
        for (unsigned int i = 0; i < m_layers.size(); i++) {
            output_mat = m_layers.at(i)->forwardWithPredict(input_mat);
            input_mat = output_mat;
        }
        return output_mat;
    }

    //void compile(std::function<> loss, std::unique_ptr<Optimizer> optimizer) {}

private:
    std::vector<std::unique_ptr<AbstLayer>> m_layers;
};

}  // namespace of MachineLearning
