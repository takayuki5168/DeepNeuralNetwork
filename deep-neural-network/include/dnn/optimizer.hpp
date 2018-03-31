/*
 * @file    optimizer.hpp
 * @brief   classes of Optimizers
 */
#pragma once

#include "dnn/abst_layer.hpp"

namespace MachineLearning
{
  
  class AbstOptimizer
  {
  public:
    explicit AbstOptimizer(const std::unique_ptr<AbstLayer>& /*layer*/) {}
    virtual void calc(std::unique_ptr<AbstLayer>& layer) = 0;
  };

  /*!
   * @class   SGD
   * @brief   class of SGD Optimizer
   */
  class SGD : public AbstOptimizer
  {
  public:
    /*!  
     * constructor
     */
    explicit SGD(const std::unique_ptr<AbstLayer>& layer, double rate = 0.1)
      : AbstOptimizer(layer)
    {
      m_rate = rate;
    }

    virtual void calc(std::unique_ptr<AbstLayer>& layer) override
    {
        layer->setWeight(layer->getWeight() - m_rate * layer->getDWeight());
    }

  private:
    double m_rate = 0.1;
  };

  /*!
   * @class   SGD
   * @brief   class of SGD Optimizer
   */
  class Momentum : public AbstOptimizer
  {
  public:
    /*!    
     * constructor
     */  
    explicit Momentum(const std::unique_ptr<AbstLayer>& layer, double learning_rate = 0.01, double momentum = 0.9)
      : AbstOptimizer(layer)
    {
      m_learning_rate = learning_rate;
      m_momentum = momentum;

      m_vel_mat = Eigen::MatrixXd::Zero(layer->getWeight().rows(), layer->getWeight().cols());      
    }

    virtual void calc(std::unique_ptr<AbstLayer>& layer) override
    {
      m_vel_mat = m_momentum * m_vel_mat - m_learning_rate * layer->getDWeight();
      layer->setWeight(layer->getWeight() + m_vel_mat);
    }

  private:
    double m_learning_rate = 0.1;
    double m_momentum = 0.9;
    Eigen::MatrixXd m_vel_mat;
  };
  
  /*!
   * @class   Adagrad
   * @brief   class of Adagrad Optimizer
   */
  class Adagrad : public AbstOptimizer
  {
  public:
    /*!  
     * constructor
     */
    explicit Adagrad(const std::unique_ptr<AbstLayer>& layer, double rate = 0.1)
      : AbstOptimizer(layer)
    {
      m_rate = rate;
      m_hadamard_mat = Eigen::MatrixXd::Zero(layer->getWeight().rows(), layer->getWeight().cols());
    }

    virtual void calc(std::unique_ptr<AbstLayer>& layer) override
    {
      Eigen::MatrixXd d_weight_mat(layer->getWeight().rows(), layer->getWeight().cols());
      for (int i = 0; i < layer->getWeight().rows(); i++){
	for (int j = 0; j < layer->getWeight().cols(); j++) {
	  m_hadamard_mat(i, j) += std::pow(layer->getDWeight()(i, j), 2);
	  d_weight_mat(i, j) = layer->getDWeight()(i, j) / std::sqrt(m_hadamard_mat(i, j) + 1e-07);
	}
      }
      layer->setWeight(layer->getWeight() - m_rate * d_weight_mat);
    }

  private:
    double m_rate = 0.1;
    Eigen::MatrixXd m_hadamard_mat;
  };


}  // namespace MachineLearning
