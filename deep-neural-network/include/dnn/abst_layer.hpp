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
     * input_num を指定しないコンストラクタ
     */
    explicit AbstLayer(int neuron_num)
        : m_neuron_num(neuron_num)
    {
        DYNAMIC_ASSERT(neuron_num < 1, "InputNum should be more than zero.");
    }
    /*
     * input_num を指定するコンストラクタ
     */
    explicit AbstLayer(int neuron_num, int input_num)
        : m_neuron_num(neuron_num), m_input_num(input_num)
    {
        DYNAMIC_ASSERT(input_num < 1, "InputNum should be more than 0.");
        initNetwork();
    }

    virtual Eigen::VectorXd forwardWithPredict(Eigen::VectorXd in_vec) { return forward(in_vec); }
    virtual Eigen::VectorXd forwardWithFit(Eigen::VectorXd in_vec) { return forward(in_vec); }
    virtual Eigen::VectorXd backwardWithFit(Eigen::VectorXd in_vec) { return backward(in_vec); }

    void setInputNum(int input_num)
    {
        DYNAMIC_ASSERT(input_num < 1, "InputNum should be more than zero.");
        m_input_num = input_num;
        initNetwork();
    }
    void setNeuronNum(int neuron_num)
    {
        DYNAMIC_ASSERT(neuron_num < 1, "NeuronNum should be more than zero.");
        m_neuron_num = neuron_num;
        initNetwork();
    }

    int getInputNum() const { return m_input_num; }
    int getNeuronNum() const { return m_neuron_num; }

protected:
    void initNetwork()
    {
        // TODO 初期化方法
        m_weight_mat.resize(m_neuron_num, m_input_num);
        for (int i = 0; i < m_neuron_num; i++) {
            for (int j = 0; j < m_input_num; j++) {
                m_weight_mat(i, j) = 0;
            }
        }

        m_out_vec.resize(m_neuron_num);
    }

    virtual Eigen::VectorXd forward(Eigen::VectorXd in_vec) { return in_vec; }
    virtual Eigen::VectorXd backward(Eigen::VectorXd in_vec) { return in_vec; }

    Eigen::MatrixXd m_weight_mat;
    int m_neuron_num = 0;
    int m_input_num = 0;

    Eigen::VectorXd m_out_vec;
};

}  // namespace of MachineLearning
