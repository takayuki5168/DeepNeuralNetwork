#pragma once

#include <random>
#include "dnn/abst_layer.hpp"

namespace MachineLearning
{

/*
class Dropout : public AbstLayer
{
public:
    explicit Dropout(double ratio)
        : AbstLayer(), m_ratio(ratio)
    {
        // m_neuron_numはすでに初期化されているか
        m_mask_vec.resize(m_neuron_num);
    }

    Eigen::MatrixXd forwardWithPredict(Eigen::MatrixXd in_mat) override { return in_mat * (1.0 - m_ratio); }
    Eigen::MatrixXd forwardWithFit(Eigen::MatrixXd in_mat) override
    {
        // TODO 乱数がうまく出ているか
        mt.seed(rnd());
        for (int i = 0; i < in_mat.size(); i++) {
            m_mask_vec(i) = (mt() % 100 <= m_ratio * 100) ? 1 : 0;
        }
        for (int i = 0; i < in_mat.size(); i++) {
            m_out_mat(i) = (m_mask_vec(i) > 0) ? in_mat(i) : 0;
        }
        return m_out_mat;
    }

private:
    Eigen::MatrixXd backward(Eigen::MatrixXd in_mat) override
    {
        for (int i = 0; i < in_mat.size(); i++) {
            in_mat(i) = (m_mask_vec(i) > 0) ? in_mat(i) : 0;
        }
        return in_mat;
    }

    const double m_ratio = 0.0;

    std::random_device rnd;
    std::mt19937 mt;
    Eigen::VectorXd m_mask_vec;
};
*/

}  // namespace of MachineLearning
