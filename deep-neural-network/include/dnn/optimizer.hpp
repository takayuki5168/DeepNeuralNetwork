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
    explicit AbstOptimizer() {}

    virtual std::function<void(AbstLayer*)> getOptFunc() const { return m_opt_func; }
  
protected:
  std::function<void(AbstLayer*)> m_opt_func;
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
    explicit SGD(double rate = 0.1)
      : AbstOptimizer()
    {
      m_rate = rate;
      m_opt_func = [this] (AbstLayer* layer)
      {
	layer->setWeight(layer->getWeight() - m_rate * layer->getDWeight());
      };
    }

private:
  double m_rate = 0.1;
};

/*!
 * @class   SGD
 * @brief   class of SGD Optimizer
 */
  /*
class Momentum : public AbstOptimizer
{
public:
    
     * constructor
    
    explicit Momentum(double rate = 0.1, double momentum = 0.9)
      : AbstOptimizer()
    {
      m_learning_rate = learning_rate;
      m_momentum = momentum;
      
      m_opt_func = [this] (std::unique_ptr<AbstLayer>& layer)
      {
	m_vel_mat = m_momentum * m_vel_mat - m_learning_rate * layer->getDWeight();
	layer->setWeight(layer->getWeight() + m_vel_mat);
      };
    }

private:
  double m_learning_rate = 0.1;
  double m_momentum = 0.9;
  Eigen::MatrixXd m_vel_mat;
};
*/

}  // namespace MachineLearning
