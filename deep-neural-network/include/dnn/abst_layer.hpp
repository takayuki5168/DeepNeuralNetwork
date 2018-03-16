/*
 *
 */
#pragma once

#include <memory>
#include "dnn/util.hpp"

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
        DYNAMIC_ASSERT(neuron_num < 1, "InputNum should be more than zero");
    }
    /*
     * input_num を指定するコンストラクタ
     */
    explicit AbstLayer(int neuron_num, int input_num)
        : m_neuron_num(neuron_num), m_input_num(input_num)
    {
        DYNAMIC_ASSERT(input_num == 0, "InputNum cannot be 0");
        initWeights();
    }

    virtual Eigen::VectorXd forward(Eigen::VectorXd /* in_val */) = 0;
    virtual Eigen::VectorXd backward(Eigen::VectorXd /* in_val */) = 0;

    void setInputNum(int input_num)
    {
        DYNAMIC_ASSERT(m_input_num != 0, "InputNum is settled automatically");
        m_input_num = input_num;
        initWeights();
    }
    void setNeuronNum(int neuron_num)
    {
        DYNAMIC_ASSERT(m_neuron_num != 0, "NeuronNum is settled automatically");
        m_neuron_num = neuron_num;
        initWeights();
    }

    int getInputNum() const { return m_input_num; }
    int getNeuronNum() const { return m_neuron_num; }

protected:
    void initWeights()
    {
        m_weights.resize(m_neuron_num, m_input_num);
        for (int i = 0; i < m_neuron_num; i++) {
            for (int j = 0; j < m_input_num; j++) {
                m_weights(i, j) = 0;
            }
        }
    }

    Eigen::MatrixXd m_weights;
    int m_neuron_num = 0;
    int m_input_num = 0;
};

}  // namespace of MachineLearning
