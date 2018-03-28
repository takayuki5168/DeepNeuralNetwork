/*
 * @file    abst_layer.hpp
 * @brief   class of AbstLayer
 */
#pragma once

#include <memory>
#include <Eigen/Geometry>
#include "dnn/assert.hpp"

namespace MachineLearning
{

class AbstLayer
{
public:
    /*!
     * @brief   constructor
     *          for Activation layer
     */
    explicit AbstLayer() {}

    /*!
     * @brief   constructor
     *          for Dense layer
     */
    explicit AbstLayer(int neuron_num)
        : m_neuron_num(neuron_num)
    {
        DYNAMIC_ASSERT(neuron_num > 0, "NeuronNum should be more than zero.");
    }
    /*!
     * @brief   constructor
     *          for the first layer
     */
    explicit AbstLayer(int neuron_num, int in_num)
        : m_neuron_num(neuron_num), m_in_num(in_num)
    {
        DYNAMIC_ASSERT(in_num > 0, "InNum should be more than 0.");
        DYNAMIC_ASSERT(neuron_num > 0, "NeuronNum should be more than 0.");
    }

    virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& /*in_mat*/, bool /*train_flag*/) = 0;
    virtual Eigen::MatrixXd backward(const Eigen::MatrixXd& /*in_mat*/) = 0;

    Eigen::MatrixXd gradDescent(const Eigen::MatrixXd& mat, const Eigen::MatrixXd& d_mat) const
    {
        return mat - 0.1 * d_mat;
    }

    void setInNum(int in_num)
    {
        DYNAMIC_ASSERT(in_num > 0, "InNum should be more than zero.");
        m_in_num = in_num;
    }
    void setNeuronNum(int neuron_num)
    {
        DYNAMIC_ASSERT(neuron_num > 0, "NeuronNum should be more than zero.");
        m_neuron_num = neuron_num;
    }

    virtual void initNetwork() {}

    int getInNum() const { return m_in_num; }
    int getNeuronNum() const { return m_neuron_num; }

protected:
    int m_neuron_num = 0;  //!< number of neuron
    int m_in_num = 0;      //!< number of input

    Eigen::MatrixXd m_weight_mat;  //!< weight matrix (m_in_num + 1) * m_neuron_num
    Eigen::MatrixXd m_d_weight_mat;

    Eigen::MatrixXd m_in_mat;  //!< input matrix
};

}  // namespace MachineLearning
