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
    explicit Momentum(const std::unique_ptr<AbstLayer>& layer, double learning_rate = 0.1, double momentum = 0.9)
      : AbstOptimizer(layer)
    {
      m_learning_rate = learning_rate;
      m_momentum = momentum;

      m_vel_mat.resize(layer->getWeight().rows(), layer->getWeight().cols());      
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


}  // namespace MachineLearning
