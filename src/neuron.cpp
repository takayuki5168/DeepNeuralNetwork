#include "include/neuron.hpp"

namespace MachineLearning
{

  Neuron::Neuron(std::function<double(double, Eigen::VectorXd, Eigen::VectorXd)> active_func, std::vector<std::shared_ptr<Neuron>> in_neuron_vec)  // = {nullptr})
    : m_active_func(active_func)
{
    m_in_neuron_vec = in_neuron_vec;
    int NumOfIn = static_cast<int>(m_in_neuron_vec.size());
    m_weigh_vec.resize(NumOfIn);
    m_in_neuron_vec.resize(NumOfIn);

    // TODO set bias and weigh random value
    m_bias = 0;
    for (int i = 0; i < NumOfIn; i++) {
        m_weigh_vec(i) = 0;
    }
}

void Neuron::calc()
{
    Eigen::VectorXd in_value_vec;
    for (unsigned int i = 0; i < m_in_neuron_vec.size(); i++) {
        in_value_vec(i) = m_in_neuron_vec.at(i)->getOutValue();
    }
    double sum = m_bias + in_value_vec.dot(m_weigh_vec);
    m_out_value = m_active_func(m_bias, in_value_vec.dot);
}

void Neuron::calc(Eigen::VectorXd in_value_vec)
{
    m_out_value = 0;
    for (unsigned int i = 0; i < in_value_vec.size(); i++) {
        m_out_value += in_value_vec(i);
    }
}

}  // namespace of MachineLearning
