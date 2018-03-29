/*!
 * @file    deep_neural_network.hpp
 * @brief   class of DeepNeuralNetwork
 */
#pragma once

#include <memory>
#include <functional>
#include "dnn/abst_layer.hpp"
#include "dnn/loss.hpp"
#include "dnn/optimizer.hpp"

//#define DEBUG_MESSAGE

namespace MachineLearning
{

/*!
 * @brief   class of DeepNeuralNetwork
 */
class DeepNeuralNetwork
{
public:
    /*!
     * @brief   constructor
     */
    explicit DeepNeuralNetwork() {}

    /*!
     * @brief   add new layer
     * @param   layer     new layer added
     */
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

        m_d_loss_func = [](const Eigen::MatrixXd& in_mat, const Eigen::MatrixXd& ans_mat) {
            return (in_mat - ans_mat);
        };
    }

    /*!
     * @brief   fit
     * @param   in_mat     input matrix of train_data
     * @param   ans_mat    answer matrix of train_data
     */
    void fit(const Eigen::MatrixXd& in_mat, const Eigen::MatrixXd& ans_mat)
    {
      std::cout << "[Fit]" << std::endl;
      
        // forward
        Eigen::MatrixXd next_in_mat = in_mat;
        for (unsigned int i = 0; i < m_layers.size(); i++) {
	  //std::cout << i << " " << next_in_mat << std::endl;
            Eigen::MatrixXd tmp_mat = m_layers.at(i)->forward(next_in_mat, true);
            next_in_mat.resize(tmp_mat.rows(), tmp_mat.cols());
            next_in_mat = tmp_mat;
        }

	double error = m_loss_func(next_in_mat, ans_mat);
	std::cout << "[Fit] error " << error << std::endl;

	
        // backward
        Eigen::MatrixXd tmp_mat = m_d_loss_func(next_in_mat, ans_mat);
	
#ifdef DEBUG_MESSAGE	
        std::cout << "[Backward]" << std::endl;
        std::cout << next_in_mat << std::endl;
        std::cout << ans_mat << std::endl;
#endif
	
	next_in_mat.resize(tmp_mat.rows(), tmp_mat.cols());
        next_in_mat = tmp_mat;

        for (unsigned int i = 0; i < m_layers.size(); i++) {
            Eigen::MatrixXd tmp_mat = m_layers.at(m_layers.size() - i - 1)->backward(next_in_mat);
	    m_opt_func(m_layers.at(m_layers.size() - i - 1));
	    
            next_in_mat.resize(tmp_mat.rows(), tmp_mat.cols());
            next_in_mat = tmp_mat;
        }
    }

    /*!
     * @brief   predict
     * @param   in_mat     input matrix to predict
     */
    Eigen::MatrixXd predict(const Eigen::MatrixXd& in_mat)
    {
        Eigen::MatrixXd next_in_mat = in_mat;
	//std::cout << "[Predict]" << std::endl;
        for (unsigned int i = 0; i < m_layers.size(); i++) {
	  //std::cout << next_in_mat << std::endl;
            Eigen::MatrixXd tmp_mat = m_layers.at(i)->forward(next_in_mat, false);
            next_in_mat.resize(tmp_mat.rows(), tmp_mat.cols());
            next_in_mat = tmp_mat;
        }
        return next_in_mat;
    }

  void compile(std::unique_ptr<AbstLoss> loss, std::unique_ptr<AbstOptimizer> optimizer)
  {
      m_loss_func = loss->getLossFunc();
      m_d_loss_func = loss->getDLossFunc();
      m_opt_func = optimizer->getOptFunc();
  }

private:
    std::vector<std::unique_ptr<AbstLayer>> m_layers;                              //!< layers
    std::function<double(Eigen::MatrixXd, Eigen::MatrixXd)> m_loss_func;  //!< loss function  
    std::function<Eigen::MatrixXd(Eigen::MatrixXd, Eigen::MatrixXd)> m_d_loss_func;  //!< derivation of loss function
  std::function<void(std::unique_ptr<AbstLayer>&)> m_opt_func;
};

}  // namespace of MachineLearning
