/*
 * @file    optimizer.hpp
 * @brief   classes of Optimizers
 */
#pragma once

namespace MachineLearning
{
  
class AbstOptimizer
{
public:
    explicit AbstOptimizer() {}

    virtual std::function<void(std::unique_ptr<AbstLayer>&)> getOptFunc() const { return m_opt_func; }
  
protected:
  std::function<void(std::unique_ptr<AbstLayer>&)> m_opt_func;
};

/*
 * @class   SGD
 * @brief   class of SGD Optimizer
 */
class SGD : public AbstOptimizer
{
public:
    /*!
     * constructor
     */
    explicit SGD()
      : AbstOptimizer()
    {
      m_opt_func = [this] (std::unique_ptr<AbstLayer>& layer)
      {
	layer->setWeight(layer->getWeight() - m_rate * layer->getDWeight());
      };
    }

private:
  double m_rate = 0.1;
};

}  // namespace MachineLearning
