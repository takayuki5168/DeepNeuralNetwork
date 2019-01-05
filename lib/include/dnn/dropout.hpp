/*!
 * @file    dropout.hpp
 * @brief   class of Dropout
 */
#pragma once

#include <random>
#include "dnn/abst_layer.hpp"

namespace MachineLearning
{

/*!
 * @class   Dropout
 * @brief   class of Dropout layer
 */
class Dropout : public AbstLayer
{
public:
    explicit Dropout(double ratio = 0.3)
        : AbstLayer(), m_ratio(ratio) {}

    /*!
     * forward propagation
     * @param in_mat       input matrix of this layer when forward propagation
     * @param train_flag   if this propagation is used to train or not
     * @return out_mat     output matrix of this layer when forward propagation
     */
    virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& in_mat, bool train_flag) override
    {
        if (train_flag) {
            m_mask_mat.resize(in_mat.rows(), in_mat.cols());
            Eigen::MatrixXd out_mat(in_mat.rows(), in_mat.cols());

            std::random_device rand;
            std::mt19937 mt;
            mt.seed(rand());
            for (int i = 0; i < in_mat.rows(); i++) {
                for (int j = 0; j < in_mat.cols(); j++) {
                    m_mask_mat(i, j) = (mt() % 100 <= m_ratio * 100) ? 1 : 0;
                    out_mat(i, j) = m_mask_mat(i, j) > 0 ? in_mat(i, j) : 0;
                }
            }
            return out_mat;
        } else {
            return in_mat * (1.0 - m_ratio);
        }
    }

    /*!
     * backward propagation
     * @param in_mat     input matrix of this layer when back propagation
     * @return out_mat   output matrix of this layer when back propagation
     */
    virtual Eigen::MatrixXd backward(const Eigen::MatrixXd& in_mat) override
    {
        Eigen::MatrixXd out_mat(in_mat.rows(), in_mat.cols());
        for (int i = 0; i < in_mat.rows(); i++) {
            for (int j = 0; j < in_mat.cols(); j++) {
                out_mat(i, j) = (m_mask_mat(i, j) > 0) ? in_mat(i, j) : 0;
            }
        }
        return out_mat;
    }

private:
    const double m_ratio = 0.0;

    std::random_device rnd;
    std::mt19937 mt;
    Eigen::VectorXd m_mask_mat;
};

}  // namespace MachineLearning
