/*
 * @file    abst_layer.hpp
 * @brief   AbstLayerクラス
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
    explicit AbstLayer() {}
    explicit AbstLayer(int neuron_num)
        : m_neuron_num(neuron_num)
    {
        DYNAMIC_ASSERT(neuron_num > 0, "NeuronNum should be more than zero.");
    }
    explicit AbstLayer(int neuron_num, int in_num)
        : m_neuron_num(neuron_num), m_in_num(in_num)
    {
        DYNAMIC_ASSERT(in_num > 0, "InNum should be more than 0.");
        DYNAMIC_ASSERT(neuron_num > 0, "NeuronNum should be more than 0.");
    }

    virtual void forwardWithPredict(const Eigen::MatrixXd& in_mat) { forward(in_mat); }
    virtual void forwardWithFit(const Eigen::MatrixXd& in_mat) { forward(in_mat); }
    virtual void backwardWithFit(const Eigen::MatrixXd& in_mat) { backward(in_mat); }

    Eigen::MatrixXd gradDescent(const Eigen::MatrixXd& mat, const Eigen::MatrixXd& d_mat)
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
    Eigen::MatrixXd getOutMat() const { return m_out_mat; }

protected:
    virtual void forward(const Eigen::MatrixXd& /*in_mat*/) {}
    virtual void backward(const Eigen::MatrixXd& /*in_mat*/) {}

    int m_neuron_num = 0;  //!< 層のニューロンの個数
    int m_in_num = 0;      //!< 入力エッジの本数

    Eigen::MatrixXd m_weight_mat;  //!< 重み行列
    Eigen::VectorXd m_bias_vec;    //!< 重み行列

    Eigen::MatrixXd m_in_mat;        //!< 入力行列
    Eigen::MatrixXd m_out_mat;       //!< 出力行列
    Eigen::MatrixXd m_back_in_mat;   //!< 入力行列
    Eigen::MatrixXd m_back_out_mat;  //!< 出力行列
};

}  // namespace of MachineLearning
