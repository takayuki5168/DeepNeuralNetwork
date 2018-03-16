/*!
 * @file    deep_neural_network.hpp
 * @brief   class of DeepNeuralNetwork
 */
#pragma once

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
    void add(std::unique_ptr<AbstLayer>&& layer)
    {
        if (not m_layers.empty()) {  // This layer is not input layer
            layer->setInputNum(m_layers.back()->getInputNum());
        }
        if (layer->getNeuronNum() == 0) {  // NeuronNum was not settled
            layer->setNeuronNum(m_layers.back()->getNeuronNum());
        }
        m_layers.push_back(std::move(layer));
    }

    void fit()
    {
    }

    Eigen::MatrixXd predict(Eigen::MatrixXd input)
    //void predict(Eigen::Matrixf)
    {
        Eigen::MatrixXd output;
        for (unsigned int i = 0; i < m_layers.size(); i++) {
            Eigen::MatrixXd output = m_layers.at(i)->forward(input);
            input = output;
        }
        return output;
    }

    //void compile(std::function<> loss, std::unique_ptr<Optimizer> optimizer) {}

private:
    std::vector<std::unique_ptr<AbstLayer>> m_layers;
};

}  // namespace of MachineLearning
