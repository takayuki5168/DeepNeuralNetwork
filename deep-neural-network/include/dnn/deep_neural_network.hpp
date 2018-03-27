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
        if (not m_layers.empty()) {            // This layer is not input layer
            if (layer->getNeuronNum() == 0) {  // NeuronNum was not settled
                layer->setInNum(m_layers.back()->getNeuronNum());
                layer->setNeuronNum(m_layers.back()->getNeuronNum());
            } else if (layer->getInNum() == 0) {  // NeuronNum was not settled
                layer->setInNum(m_layers.back()->getNeuronNum());
            }
            DYNAMIC_ASSERT(layer->getNeuronNum() > 0, "NeuronNum should be more than zero.");
        } else {                               // This layer is input layer
            if (layer->getNeuronNum() == 0) {  // NeuronNum was not settled
                layer->setInNum(m_layers.back()->getNeuronNum());
                layer->setNeuronNum(m_layers.back()->getNeuronNum());
            }
        }
        layer->initNetwork();
        m_layers.push_back(std::move(layer));

        d_loss_func = [](const Eigen::MatrixXd& in_mat, const Eigen::MatrixXd& ans_mat) {
            return (in_mat - ans_mat);
        };
    }

    void fit(const Eigen::MatrixXd& in_mat, const Eigen::MatrixXd& ans_mat)
    {
        // forward
        Eigen::MatrixXd next_in_mat = in_mat;
        for (unsigned int i = 0; i < m_layers.size(); i++) {
            Eigen::MatrixXd tmp_mat = m_layers.at(i)->forward(next_in_mat, true);
            next_in_mat.resize(tmp_mat.rows(), tmp_mat.cols());
            next_in_mat = tmp_mat;
        }

        // backward
        Eigen::MatrixXd tmp_mat = d_loss_func(next_in_mat, ans_mat);
        std::cout << "[Backward]" << std::endl;
        std::cout << next_in_mat << std::endl;
        std::cout << ans_mat << std::endl;

        next_in_mat.resize(tmp_mat.rows(), tmp_mat.cols());
        next_in_mat = tmp_mat;

        for (unsigned int i = 0; i < m_layers.size(); i++) {
            Eigen::MatrixXd tmp_mat = m_layers.at(m_layers.size() - i - 1)->backward(next_in_mat);
            next_in_mat.resize(tmp_mat.rows(), tmp_mat.cols());
            next_in_mat = tmp_mat;
        }
    }

    Eigen::MatrixXd predict(const Eigen::MatrixXd& in_mat)
    {
        Eigen::MatrixXd next_in_mat = in_mat;
        for (unsigned int i = 0; i < m_layers.size(); i++) {
            Eigen::MatrixXd tmp_mat = m_layers.at(i)->forward(next_in_mat, false);
            next_in_mat.resize(tmp_mat.rows(), tmp_mat.cols());
            next_in_mat = tmp_mat;
        }
        return next_in_mat;
    }

    //void compile(std::function<> loss, std::unique_ptr<Optimizer> optimizer) {}

private:
    std::vector<std::unique_ptr<AbstLayer>> m_layers;
    std::function<Eigen::MatrixXd(Eigen::MatrixXd, Eigen::MatrixXd)> d_loss_func;
};

}  // namespace of MachineLearning
