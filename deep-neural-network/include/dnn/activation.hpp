/*
 * @file    activation.hpp
 * @brief   Activationクラス群
 */
#pragma once

#include "dnn/abst_layer.hpp"
#include "dnn/math_util.hpp"

namespace MachineLearning
{
/*
class Identify : public AbstLayer
{
public:
    explicit Identify() : AbstLayer() {}

    Eigen::MatrixXd forward(Eigen::MatrixXd in_mat) override { return in_mat; }
    Eigen::MatrixXd backward(Eigen::MatrixXd in_mat) override { return in_mat; }

private:
};

class Step : public AbstLayer
{
public:
    explicit Step() : AbstLayer() {}

    Eigen::MatrixXd forward(Eigen::MatrixXd in_mat) override
    {
        Eigen::MatrixXd out_mat;
        for (int i = 0; i < in_mat.size(); i++) {
            out_mat(i) = (in_mat(i) > 0) ? 1 : 0;
        }
        return out_mat;
    }
    // TODO
    Eigen::MatrixXd backward(Eigen::MatrixXd in_mat) override
    {
        return in_mat;
    }

private:
};
*/

/*
 * @brief   Softmax
 */
/*
class Softmax : public AbstLayer
{
public:
    explicit Softmax() : AbstLayer() {}

private:
    Eigen::MatrixXd forward(const Eigen::MatrixXd& in_mat) override
    {
        m_in_mat = in_mat;

        // calc max vector of each column of in_mat
        m_out_mat.resize(in_mat.rows(), in_mat.cols());
        Eigen::VectorXd max_vec = in_mat.block(0, 0, 1, in_mat.cols());
        for (int i = 0; i < in_mat.cols(); i++) {
            // CHECK
            for (int j = 0; j < in_mat.rows(); j++) {
                max_vec(i) = (max_vec(i) > in_mat(j, i)) ? max_vec(i) : in_mat(j, i);
            }
        }

        // calc exponential mat
        Eigen::MatrixXd exp_mat;
        exp_mat.resize(in_mat.rows(), in_mat.cols());
        for (int i = 0; i < in_mat.cols(); i++) {
            for (int j = 0; j < in_mat.rows(); j++) {
                exp_mat(j, i) = std::exp(in_mat(j, i) - max_vec(i));
            }

            double sum_val = 0.0;
            for (int j = 0; j < in_mat.rows(); j++) {
                sum_val += exp_mat(j, i);
            }

            for (int j = 0; j < in_mat.rows(); j++) {
                exp_mat(j, i) = exp_mat(j, i) / sum_val;
            }
        }
        m_out_mat = exp_mat;
    }
    void backward(const Eigen::MatrixXd& back_in_mat) override
    {
        m_back_in_mat = back_in_mat;

        Eigen::VectorXd sum_vec = Eigen::VectorXd::Zero(back_in_mat.cols());
        m_back_out_mat.resize(back_in_mat.rows(), back_in_mat.cols());
        for (int i = 0; i < back_in_mat.cols(); i++) {
            for (int j = 0; j < back_in_mat.rows(); j++) {
                sum_vec(i) += m_out_mat(j, i) * back_in_mat(j, i);
            }
        }

        for (int i = 0; i < back_in_mat.cols(); i++) {
            for (int j = 0; j < back_in_mat.rows(); j++) {
                m_back_out_mat(j, i) = m_out_mat(j, i) * (back_in_mat(j, i) - sum_vec(i));
            }
        }
    }
};

class Sigmoid : public AbstLayer
{
public:
    explicit Sigmoid() : AbstLayer() {}

private:
    void forward(const Eigen::MatrixXd& in_mat) override
    {
        m_in_mat = in_mat;

        m_out_mat.resize(in_mat.rows(), in_mat.cols());
        for (int i = 0; i < in_mat.size(); i++) {
            for (int j = 0; j < in_mat.rows(); j++) {
                m_out_mat(j, i) = 1.0 / (1.0 + std::exp(-in_mat(j, i)));
            }
        }
    }
    void backward(const Eigen::MatrixXd& in_mat) override
    {
        m_back_in_mat = in_mat;

        m_back_out_mat.resize(in_mat.rows(), in_mat.cols());
        for (int i = 0; i < in_mat.size(); i++) {
            for (int j = 0; j < in_mat.rows(); j++) {
                m_back_out_mat(i) = in_mat(j, i) * (1.0 - m_out_mat(j, i)) * m_out_mat(j, i);
            }
        }
    }
};

class Relu : public AbstLayer
{
public:
    explicit Relu() : AbstLayer() {}

private:
    void forward(const Eigen::MatrixXd& in_mat) override
    {
        m_in_mat = in_mat;

        m_out_mat.resize(in_mat.rows(), in_mat.cols());
        for (int i = 0; i < in_mat.cols(); i++) {
            for (int j = 0; j < in_mat.rows(); j++) {
                m_back_out_mat(j, i) = (in_mat(j, i) > 0) ? in_mat(j, i) : 0;
            }
        }
    }
    void backward(const Eigen::MatrixXd& in_mat) override
    {
        m_back_in_mat = in_mat;

        m_back_out_mat.resize(in_mat.rows(), in_mat.cols());
        for (int i = 0; i < in_mat.cols(); i++) {
            for (int j = 0; j < in_mat.rows(); j++) {
                m_back_out_mat(j, i) = (in_mat(j, i) > 0) ? in_mat(j, i) : 0;
            }
        }
    }
};
*/

}  // namespace of MachineLearning
