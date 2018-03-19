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
            layer->setInNum(m_layers.back()->getInNum());
        }
        if (layer->getNeuronNum() == 0) {  // NeuronNum was not settled
            layer->setInNum(m_layers.back()->getNeuronNum());
            layer->setNeuronNum(m_layers.back()->getNeuronNum());
        }
        m_layers.push_back(std::move(layer));
    }

    void fit(const Eigen::MatrixXd& in_mat)
    {
        Eigen::MatrixXd next_in_mat = in_mat;
        for (unsigned int i = 0; i < m_layers.size(); i++) {
            m_layers.at(i)->forwardWithFit(next_in_mat);
            next_in_mat = m_layers.at(i)->getOutMat();
        }

        //input_mat = ;
        /*
        for (unsigned int i = 0; i < m_layers.size(); i++) {
            output_mat = m_layers.at(m_layers.size() - i - 1)->backwardWithFit(input_mat);
            input_mat = output_mat;
        }
        */
    }

    Eigen::MatrixXd predict(const Eigen::MatrixXd& in_mat)
    {
        Eigen::MatrixXd next_in_mat = in_mat;
        for (unsigned int i = 0; i < m_layers.size(); i++) {
            m_layers.at(i)->forwardWithPredict(in_mat);
            next_in_mat = m_layers.at(i)->getOutMat();
        }
        return next_in_mat;
    }

    //void compile(std::function<> loss, std::unique_ptr<Optimizer> optimizer) {}

private:
    std::vector<std::unique_ptr<AbstLayer>> m_layers;
};

}  // namespace of MachineLearning
