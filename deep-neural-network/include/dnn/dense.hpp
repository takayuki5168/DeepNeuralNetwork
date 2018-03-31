/*!
 * @file    dense.hpp
 * @brief   class of Dense Layer
 */
#pragma once

#include <random>
#include "dnn/abst_layer.hpp"

namespace MachineLearning
{

  /*!
   * @class   Dense
   * @brief   class of Dense Layer
   */
  class Dense : public AbstLayer
  {
  public:
    /*!
     * constructor
     * @param neuron_num   number of neuron of this layer
     * @note   for except the first layer
     */
    explicit Dense(int neuron_num)
      : AbstLayer(neuron_num) {}
    /*!
     * constructor
     * @param neuron_num   number of neuron of this layer
     * @param in_num       number of input of this layer
     * @note   for the first layer
     */
    explicit Dense(int neuron_num, int in_num)
      : AbstLayer(neuron_num, in_num) {}

    /*!
     * resize weight and d_weight matrix 
     */
    virtual void initNetwork() override
    {
      m_weight_mat.resize(m_in_num + 1, m_neuron_num);
      m_d_weight_mat.resize(m_weight_mat.rows(), m_weight_mat.cols());
	
      std::random_device rand;
      std::mt19937 mt;
      mt.seed(rand());
      for (int i = 0; i < m_weight_mat.rows(); i++) {
	for (int j = 0; j < m_weight_mat.cols(); j++) {
	  m_weight_mat(i, j) = (static_cast<double>(mt() % 1000) - 500) / 5000;
	}
      }
#ifdef DEBUG_MESSAGE	
      std::cout << "[Init Weight]" << std::endl;
      std::cout << m_weight_mat << std::endl;
#endif
    }

    /*!
     * forward propagation
     * @param in_mat   input matrix
     */
    virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& in_mat, bool /*train_flag*/) override
    {
      // init X
      m_in_mat.resize(in_mat.rows(), in_mat.cols() + 1);
      m_in_mat = Eigen::MatrixXd::Ones(in_mat.rows(), in_mat.cols() + 1);
      m_in_mat.block(0, 1, in_mat.rows(), in_mat.cols()) = in_mat;

      // Y = XW
      Eigen::MatrixXd out_mat = m_in_mat * m_weight_mat;

#ifdef DEBUG_MESSAGE
      std::cout << "[Forward]" << std::endl;
      std::cout << in_mat << std::endl;
      std::cout << m_in_mat << std::endl;
      std::cout << m_weight_mat << std::endl;
      std::cout << "[Forward] out_mat" << std::endl;
      std::cout << out_mat << std::endl;
      std::cout << "[Forward] end" << std::endl;
#endif

      return out_mat;
    }

    virtual Eigen::MatrixXd backward(const Eigen::MatrixXd& in_mat) override
    {
      m_d_weight_mat = m_in_mat.transpose() * in_mat;
      return (in_mat * m_weight_mat.transpose()).block(0, 1, m_in_mat.rows(), m_in_mat.cols() - 1);
    }

  private:
  };

}  // namespace MachineLearning
