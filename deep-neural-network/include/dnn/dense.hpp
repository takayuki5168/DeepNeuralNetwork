#pragma once

#include "dnn/abst_layer.hpp"

namespace MachineLearning
{

class Dense : public AbstLayer
{
public:
    explicit Dense(int neuron_num)
        : AbstLayer(neuron_num) { m_bias_vec.resize(neuron_num); }
    explicit Dense(int neuron_num, int in_num)
        : AbstLayer(neuron_num, in_num) { m_bias_vec.resize(neuron_num); }

    virtual void initNetwork()
    {
        m_weight_mat.resize(m_neuron_num, m_in_num);
        m_bias_vec.resize(m_neuron_num);
        for (int i = 0; i < m_neuron_num; i++) {
            for (int j = 0; j < m_in_num; j++) {
                m_weight_mat(i, j) = 0;
            }
            m_bias_vec(i) = 0;
        }
    }

private:
    void forward(const Eigen::MatrixXd& in_mat) override
    {
        m_in_mat = in_mat;

        // transform bias vector to bias matrix
        Eigen::MatrixXd bias_mat(m_bias_vec.size(), in_mat.cols());
        for (int i = 0; i < bias_mat.cols(); i++) {
            //bias_mat.block(0, i, in_mat.cols(), 1) = m_bias_vec;
            bias_mat.col(i) = m_bias_vec;
        }

        m_out_mat = in_mat * m_weight_mat + bias_mat;
    }
    void backward(const Eigen::MatrixXd& in_mat) override
    {
        m_back_in_mat = in_mat;

        // update weight matrix
        Eigen::MatrixXd d_weight_mat = m_in_mat.transpose() * in_mat;
        gradDescent(m_weight_mat, d_weight_mat);

        // update bias vector
        Eigen::MatrixXd d_bias_mat = in_mat;
        gradDescent(m_weight_mat, d_weight_mat);

        m_back_out_mat = in_mat * m_weight_mat.transpose();
    }

    //Eigen::VectorXd m_bias_vec;  //!< bias vector
};

}  // namespace of MachineLearning
