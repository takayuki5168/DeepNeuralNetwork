/*
 *
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
    /*
     * in_num を指定しないコンストラクタ
     */
    explicit AbstLayer(int neuron_num)
        : m_neuron_num(neuron_num)
    {
        DYNAMIC_ASSERT(neuron_num < 1, "inNum should be more than zero.");
    }
    /*
     * in_num を指定するコンストラクタ
     */
    explicit AbstLayer(int neuron_num, int in_num)
        : m_neuron_num(neuron_num), m_in_num(in_num)
    {
        DYNAMIC_ASSERT(in_num < 1, "inNum should be more than 0.");
        initNetwork();
    }

    virtual void forwardWithPredict(const Eigen::MatrixXd& in_mat) { forward(in_mat); }
    virtual void forwardWithFit(const Eigen::MatrixXd& in_mat) { forward(in_mat); }
    virtual void backwardWithFit(const Eigen::MatrixXd& in_mat) { backward(in_mat); }

    void gradDescent(Eigen::MatrixXd& mat, const Eigen::MatrixXd& d_mat)
    {
        mat = mat - 0.1 * d_mat;
    }

    void setInNum(int in_num)
    {
        DYNAMIC_ASSERT(in_num < 1, "inNum should be more than zero.");
        m_in_num = in_num;
        initNetwork();
    }
    void setNeuronNum(int neuron_num)
    {
        DYNAMIC_ASSERT(neuron_num < 1, "NeuronNum should be more than zero.");
        m_neuron_num = neuron_num;
        initNetwork();
    }

    int getInNum() const { return m_in_num; }
    int getNeuronNum() const { return m_neuron_num; }
    Eigen::MatrixXd getOutMat() const { return m_out_mat; }

protected:
    void initNetwork()
    {
        // TODO 初期化方法
        m_weight_mat.resize(m_neuron_num, m_in_num);
        for (int i = 0; i < m_neuron_num; i++) {
            for (int j = 0; j < m_in_num; j++) {
                m_weight_mat(i, j) = 0;
            }
        }

        //m_out_mat.resize(m_neuron_num);
    }

    virtual void forward(const Eigen::MatrixXd& /*in_mat*/) {}
    virtual void backward(const Eigen::MatrixXd& /*in_mat*/) {}

    Eigen::MatrixXd m_weight_mat;  //!< 重み行列
    int m_neuron_num = 0;          //!< 層のニューロンの個数
    int m_in_num = 0;              //!< 入力エッジの本数

    Eigen::MatrixXd m_in_mat;        //!< 入力行列
    Eigen::MatrixXd m_out_mat;       //!< 出力行列
    Eigen::MatrixXd m_back_in_mat;   //!< 入力行列
    Eigen::MatrixXd m_back_out_mat;  //!< 出力行列
};

}  // namespace of MachineLearning
