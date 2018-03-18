#pragma once

#include <random>
#include "dnn/abst_layer.hpp"

namespace MachineLearning
{

class Dropout : public AbstLayer
{
public:
    explicit Dropout(double ratio)
        : AbstLayer(), m_ratio(ratio)
    {
        // m_neuron_numはすでに初期化されているか
        m_mask_vec.resize(m_neuron_num);
    }

    Eigen::VectorXd forwardWithPredict(Eigen::VectorXd in_vec) override { return in_vec * (1.0 - m_ratio); }
    Eigen::VectorXd forwardWithFit(Eigen::VectorXd in_vec) override
    {
        // TODO 乱数がうまく出ているか
        mt.seed(rnd());
        for (int i = 0; i < in_vec.size(); i++) {
            m_mask_vec(i) = (mt() % 100 <= m_ratio * 100) ? 1 : 0;
        }
        for (int i = 0; i < in_vec.size(); i++) {
            m_out_vec(i) = (m_mask_vec(i) > 0) ? in_vec(i) : 0;
        }
        return m_out_vec;
    }

private:
    Eigen::VectorXd backward(Eigen::VectorXd in_vec) override
    {
        for (int i = 0; i < in_vec.size(); i++) {
            in_vec(i) = (m_mask_vec(i) > 0) ? in_vec(i) : 0;
        }
        return in_vec;
    }

    const double m_ratio = 0.0;

    std::random_device rnd;
    std::mt19937 mt;
    Eigen::VectorXd m_mask_vec;
};

}  // namespace of MachineLearning
