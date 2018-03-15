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
    explicit AbstLayer(int neuron_num, int input_num = 0)
        : m_neuron_num(neuron_num), m_input_num(input_num)
    {
        DYNAMIC_ASSERT(input_num == 0, "Input Num cannot be 0");
        initWeights();
    }

    virtual Eigen::VectorXf forward(Eigen::VectorXf /* in_val */) = 0;
    virtual Eigen::VectorXf backward(Eigen::VectorXf /* in_val */) = 0;
    void setInputNum(int input_num)
    {
        DYNAMIC_ASSERT(m_input_num == 0, "Input Num is settled automatically");
        m_input_num = input_num;
        initWeights();
    }
    int getInputNum() const { return m_input_num; }

protected:
    using ActFunc = std::function<std::vector<float>(std::vector<float>)>;

    void initWeights()
    {
        m_weights.resize(m_neuron_num);
        for (int i = 0; i < m_neuron_num; i++) {
            m_weights.at(i).resize(m_input_num);
        }
    }

    std::vector<std::vector<float>> m_weights;
    const int m_neuron_num = 0;
    int m_input_num = 0;
};

}  // namespace of MachineLearning
