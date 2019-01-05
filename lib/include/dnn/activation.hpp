/*
 * @file    activation.hpp
 * @brief   classes of Activation
 */
#pragma once

#include "dnn/abst_layer.hpp"

namespace MachineLearning
{
/*!
 * @class   Softmax
 * @brief   class of Softmax Activation
 */
class Softmax : public AbstLayer
{
public:
    explicit Softmax() : AbstLayer() {}

    /*!
     * forward propagation
     * @param in_mat       input matrix of this layer when forward propagation
     * @param train_flag   if this propagation is used to train or not
     * @return out_mat     output matrix of this layer when forward propagation
     */
    virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& in_mat, bool /*train_flag*/) override
    {
        m_in_mat.resize(in_mat.rows(), in_mat.cols());
        m_in_mat = in_mat;

        // calc max vector of each column of in_mat
        Eigen::VectorXd max_vec = in_mat.block(0, 0, in_mat.rows(), 1);
        for (int i = 0; i < in_mat.cols(); i++) {
            for (int j = 0; j < in_mat.rows(); j++) {
                max_vec(i) = (max_vec(i) > in_mat(j, i)) ? max_vec(i) : in_mat(j, i);
            }
        }

        // calc exponential mat
        Eigen::MatrixXd exp_mat;
        exp_mat.resize(in_mat.rows(), in_mat.cols());
        for (int j = 0; j < in_mat.rows(); j++) {
            for (int i = 0; i < in_mat.cols(); i++) {
                exp_mat(j, i) = std::exp(in_mat(j, i) - max_vec(i));
            }

            double sum_val = 0.0;
            for (int i = 0; i < in_mat.cols(); i++) {
                sum_val += exp_mat(j, i);
            }

            for (int i = 0; i < in_mat.cols(); i++) {
                exp_mat(j, i) = exp_mat(j, i) / sum_val;
            }
        }
        return exp_mat;
    }

    /*!
     * back propagation
     * @param in_mat   input matrix of this layer when back propagation
     * @return out_mat   output matrix of this layer when back propagation
     */
    virtual Eigen::MatrixXd backward(const Eigen::MatrixXd& in_mat) override
    {
        Eigen::VectorXd sum_vec = Eigen::VectorXd::Zero(in_mat.cols());
        for (int i = 0; i < in_mat.cols(); i++) {
            for (int j = 0; j < in_mat.rows(); j++) {
                sum_vec(i) += m_in_mat(j, i) * in_mat(j, i);
            }
        }

        Eigen::MatrixXd out_mat(in_mat.rows(), in_mat.cols());
        for (int i = 0; i < in_mat.cols(); i++) {
            for (int j = 0; j < in_mat.rows(); j++) {
                out_mat(j, i) = m_in_mat(j, i) * (in_mat(j, i) - sum_vec(i));
            }
        }
        return out_mat;
    }

private:
};

/*!
 * @class   Sigmoid
 * @brief   class of Sigmoid Activation
 */
class Sigmoid : public AbstLayer
{
public:
    /*!
     * constructor
     */
    explicit Sigmoid() : AbstLayer() {}

    /*!
     * forward propagation
     * @param in_mat       input matrix of this layer when forward propagation
     * @param train_flag   if this propagation is used to train or not
     * @return out_mat     output matrix of this layer when forward propagation
     */
    virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& in_mat, bool /*train_flag*/) override
    {
        m_in_mat.resize(in_mat.rows(), in_mat.cols());
        m_in_mat = in_mat;

        Eigen::MatrixXd out_mat(in_mat.rows(), in_mat.cols());
        for (int i = 0; i < in_mat.rows(); i++) {
            for (int j = 0; j < in_mat.cols(); j++) {
                out_mat(i, j) = 1. / (1. + std::exp(-in_mat(i, j)));
            }
        }
        return out_mat;
    }

    /*!
     * back propagation
     * @param in_mat   input matrix of this layer when back propagation
     * @return out_mat   output matrix of this layer when back propagation
     * @return out_mat     output matrix of this layer when back propagation
     */
    virtual Eigen::MatrixXd backward(const Eigen::MatrixXd& in_mat) override
    {
        Eigen::MatrixXd out_mat(in_mat.rows(), in_mat.cols());
        for (int i = 0; i < in_mat.rows(); i++) {
            for (int j = 0; j < in_mat.cols(); j++) {
                double sigmoid_val = 1. / (1. + std::exp(-m_in_mat(i, j)));
                out_mat(i, j) = in_mat(i, j) * sigmoid_val * (1. - sigmoid_val);
            }
        }
        return out_mat;
    }

private:
};

/*!
 * @class   ReLU
 * @brief   class of ReLU Activation
 */
class ReLU : public AbstLayer
{
public:
    /*!
     * constructor
     */
    explicit ReLU() : AbstLayer() {}

    /*!
     * forward propagation
     * @param in_mat       input matrix of this layer when forward propagation
     * @param train_flag   if this propagation is used to train or not
     * @return out_mat     output matrix of this layer when forward propagation
     */
    virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& in_mat, bool /*train_flag*/) override
    {
        m_in_mat.resize(in_mat.rows(), in_mat.cols());
        m_in_mat = in_mat;

        Eigen::MatrixXd out_mat(in_mat.rows(), in_mat.cols());
        for (int i = 0; i < in_mat.rows(); i++) {
            for (int j = 0; j < in_mat.cols(); j++) {
                out_mat(i, j) = (in_mat(i, j) > 0) ? in_mat(i, j) : 0.;
            }
        }
        return out_mat;
    }

    /*!
     * back propagation
     * @param in_mat   input matrix of this layer when back propagation
     * @return out_mat   output matrix of this layer when back propagation
     * @return out_mat     output matrix of this layer when back propagation
     */
    virtual Eigen::MatrixXd backward(const Eigen::MatrixXd& in_mat) override
    {
        Eigen::MatrixXd out_mat(in_mat.rows(), in_mat.cols());
        for (int i = 0; i < in_mat.rows(); i++) {
            for (int j = 0; j < in_mat.cols(); j++) {
                out_mat(i, j) = (m_in_mat(i, j) > 0) ? in_mat(i, j) : 0.;
            }
        }
        return out_mat;
    }

private:
};

}  // namespace MachineLearning
