/*
 * @file    abst_layer.hpp
 * @brief   class of AbstLayer
 */
#pragma once

#include <memory>
#include <Eigen/Geometry>
#include "dnn/runtime_assert.hpp"

namespace MachineLearning
{

/*!
 * @class   AbstLayer
 * @brief   class of AbstLayer
 */
class AbstLayer
{
public:
    /*!
     * constructor
     * @note   for Activation layer
     */
    explicit AbstLayer() {}

    /*!
     * constructor
     * @note   for Dense layer
     */
    explicit AbstLayer(int neuron_num)
        : m_neuron_num(neuron_num)
    {
        RUNTIME_ASSERT(neuron_num > 0, "NeuronNum should be more than zero.");
    }
    /*!
     * constructor
     * @note   for the first layer of network
     */
    explicit AbstLayer(int neuron_num, int in_num)
        : m_neuron_num(neuron_num), m_in_num(in_num)
    {
        RUNTIME_ASSERT(in_num > 0, "InNum should be more than 0.");
        RUNTIME_ASSERT(neuron_num > 0, "NeuronNum should be more than 0.");
    }

    virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& in_mat, bool train_flag) = 0;
    virtual Eigen::MatrixXd backward(const Eigen::MatrixXd& in_mat) = 0;

    void setInNum(int in_num)
    {
        RUNTIME_ASSERT(in_num > 0, "InNum should be more than zero.");
        m_in_num = in_num;
    }
    void setNeuronNum(int neuron_num)
    {
        RUNTIME_ASSERT(neuron_num > 0, "NeuronNum should be more than zero.");
        m_neuron_num = neuron_num;
    }

    virtual void initNetwork() {}

    int getInNum() const { return m_in_num; }
    int getNeuronNum() const { return m_neuron_num; }

    Eigen::MatrixXd getWeight() const { return m_weight_mat; }
    Eigen::MatrixXd getDWeight() const { return m_d_weight_mat; }
    void setWeight(Eigen::MatrixXd weight_mat) { m_weight_mat = weight_mat; }

protected:
    int m_neuron_num = 0;  //!< number of neuron
    int m_in_num = 0;      //!< number of input

    Eigen::MatrixXd m_weight_mat;  //!< weight matrix (m_in_num + 1) * m_neuron_num
    Eigen::MatrixXd m_d_weight_mat; //!< derive weight matrix (m_in_num + 1) * m_neuron_num

    Eigen::MatrixXd m_in_mat;  //!< input matrix of this layer when forward propagation
};

}  // namespace MachineLearning
