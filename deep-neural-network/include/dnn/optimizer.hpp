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
   * @class   MomentumSGD
   * @brief   class of MomentumSGD Optimizer
   */
  class MomentumSGD : public AbstOptimizer
  {
  public:
    /*!    
     * constructor
     */  
    explicit MomentumSGD(const std::unique_ptr<AbstLayer>& layer, double learning_rate = 0.01, double momentum = 0.9)
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
   * @class   AdaGrad
   * @brief   class of AdaGrad Optimizer
   */
  class AdaGrad : public AbstOptimizer
  {
  public:
    /*!  
     * constructor
     */
    explicit AdaGrad(const std::unique_ptr<AbstLayer>& layer, double rate = 0.1)
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
  

  /*!
   * @class   RMSprop
   * @brief   class of RMSprop Optimizer
   */
  class RMSprop : public AbstOptimizer
  {
  public:
    /*!  
     * constructor
     */
    explicit RMSprop(const std::unique_ptr<AbstLayer>& layer, double rate = 0.001, double alpha = 0.99)
      : AbstOptimizer(layer), m_alpha(alpha)
    {
      m_rate = rate;
      m_hadamard_mat = Eigen::MatrixXd::Zero(layer->getWeight().rows(), layer->getWeight().cols());
    }

    virtual void calc(std::unique_ptr<AbstLayer>& layer) override
    {
      Eigen::MatrixXd d_weight_mat(layer->getWeight().rows(), layer->getWeight().cols());
      for (int i = 0; i < layer->getWeight().rows(); i++){
	for (int j = 0; j < layer->getWeight().cols(); j++) {
	  m_hadamard_mat(i, j) = m_alpha * m_hadamard_mat(i, j) + (1 - m_alpha) * std::pow(layer->getDWeight()(i, j), 2);
	  d_weight_mat(i, j) = layer->getDWeight()(i, j) / (std::sqrt(m_hadamard_mat(i, j)) + 1e-08);
	}
      }
      layer->setWeight(layer->getWeight() - m_rate * d_weight_mat);
    }

  private:
    double m_rate = 0.001;
    double m_alpha = 0.99;
    Eigen::MatrixXd m_hadamard_mat;
  };
  
  /*!
   * @class   AdaDelta
   * @brief   class of AdaDelta Optimizer
   */
  class AdaDelta : public AbstOptimizer
  {
  public:
    /*!  
     * constructor
     */
    explicit AdaDelta(const std::unique_ptr<AbstLayer>& layer, double rho = 0.95)
      : AbstOptimizer(layer), m_rho(rho)
    {
      m_h_mat = Eigen::MatrixXd::Zero(layer->getWeight().rows(), layer->getWeight().cols());
      m_v_mat = Eigen::MatrixXd::Zero(layer->getWeight().rows(), layer->getWeight().cols());      
      m_s_mat = Eigen::MatrixXd::Zero(layer->getWeight().rows(), layer->getWeight().cols());      
    }

    virtual void calc(std::unique_ptr<AbstLayer>& layer) override
    {
      for (int i = 0; i < layer->getWeight().rows(); i++){
	for (int j = 0; j < layer->getWeight().cols(); j++) {
	  m_h_mat(i, j) = m_rho * m_h_mat(i, j) + (1 - m_rho) * std::pow(layer->getDWeight()(i, j), 2);
          m_v_mat(i, j) = std::sqrt(m_s_mat(i, j) + 1e-6) / std::sqrt(m_h_mat(i, j) + 1e-6) * layer->getDWeight()(i, j);
	  m_s_mat(i, j) = m_rho * m_s_mat(i, j) + (1 - m_rho) * std::pow(m_v_mat(i, j), 2);
	}
      }
      layer->setWeight(layer->getWeight() - m_v_mat);
    }

  private:
    double m_rho = 0.95;
    
    Eigen::MatrixXd m_h_mat;
    Eigen::MatrixXd m_s_mat;    
    Eigen::MatrixXd m_v_mat;    
  };

    /*!
   * @class   Adam
   * @brief   class of Adam Optimizer
   */
  class Adam : public AbstOptimizer
  {
  public:
    /*!  
     * constructor
     */
    explicit Adam(const std::unique_ptr<AbstLayer>& layer, double rate = 0.001, double alpha = 0.9, double beta = 0.999)
      : AbstOptimizer(layer), m_alpha(alpha), m_beta(beta)
    {
      m_rate = rate;
      m_m_mat = Eigen::MatrixXd::Zero(layer->getWeight().rows(), layer->getWeight().cols());
      m_v_mat = Eigen::MatrixXd::Zero(layer->getWeight().rows(), layer->getWeight().cols());      
    }

    virtual void calc(std::unique_ptr<AbstLayer>& layer) override
    {
      m_m_mat = m_alpha * m_m_mat + (1 - m_alpha) * layer->getDWeight();
      
      for (int i = 0; i < layer->getWeight().rows(); i++){
	for (int j = 0; j < layer->getWeight().cols(); j++) {
	  m_v_mat(i, j) = m_beta * m_v_mat(i, j) + (1 - m_beta) * std::pow(layer->getDWeight()(i, j), 2);
	}
      }
      
      Eigen::MatrixXd d_weight_mat(layer->getWeight().rows(), layer->getWeight().cols());      
      for (int i = 0; i < layer->getWeight().rows(); i++){
	for (int j = 0; j < layer->getWeight().cols(); j++) {
	  d_weight_mat(i, j) = m_m_mat(i, j) / (1 - m_alpha) / (m_v_mat(i, j) / (1 - m_beta) + 1e-7);
	}
      }

      layer->setWeight(layer->getWeight() - m_rate * d_weight_mat);
    }

  private:
    double m_rate = 0.001;
    double m_alpha = 0.9;
    double m_beta = 0.999;
    
    Eigen::MatrixXd m_m_mat;
    Eigen::MatrixXd m_v_mat;    
  };

}  // namespace MachineLearning
