#include "include/neuron.hpp"

namespace MachineLearning
{

void Neuron::initNeuron(int in_num, std::function<double(double)> func)
{
    m_in_num = in_num;

    m_in_value.resize(m_in_num);
    m_weigh.resize(m_in_num);

    // init random value
    m_bias = 0;
    for (int i = 0; i < m_in_num; i++) {
        m_weigh(i) = i;
    }

    m_active_func = func;
}

double Neuron::calc()
{
    double sum = m_bias + m_in_value.dot(m_weigh);
    return m_active_func(sum);
}

}  // namespace of MachineLearning
