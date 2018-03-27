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
class Softmax : public AbstLayer
{
public:
    explicit Softmax() : AbstLayer() {}

    Eigen::MatrixXd forward(const Eigen::MatrixXd& in_mat, bool train_flag) override
    {
        std::cout << "[Softmax] forward" << std::endl;
        std::cout << in_mat << std::endl;

        m_in_mat.resize(in_mat.rows(), in_mat.cols());
        m_in_mat = in_mat;

        // calc max vector of each column of in_mat
        Eigen::VectorXd max_vec = in_mat.block(0, 0, in_mat.rows(), 1);
        for (int i = 0; i < in_mat.cols(); i++) {
            // CHECK
            for (int j = 0; j < in_mat.rows(); j++) {
                max_vec(i) = (max_vec(i) > in_mat(j, i)) ? max_vec(i) : in_mat(j, i);
            }
        }
        std::cout << max_vec << std::endl;

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
        std::cout << exp_mat << std::endl;
        return exp_mat;
    }

    Eigen::MatrixXd backward(const Eigen::MatrixXd& in_mat) override
    {
        return in_mat;
        /*
        Eigen::VectorXd sum_vec = Eigen::VectorXd::Zero(in_mat.cols());
        Eigen::MatrixXd out_mat(in_mat.rows(), in_mat.cols());
        for (int i = 0; i < in_mat.cols(); i++) {
            for (int j = 0; j < in_mat.rows(); j++) {
                sum_vec(i) += in_mat(j, i) * in_mat(j, i);
            }
        }

        for (int i = 0; i < in_mat.cols(); i++) {
            for (int j = 0; j < in_mat.rows(); j++) {
                out_mat(j, i) = m_in_mat(j, i) * (in_mat(j, i) - sum_vec(i));
            }
        }
        */
    }

private:
};

/*
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

*/
class Relu : public AbstLayer
{
public:
    explicit Relu() : AbstLayer() {}

    Eigen::MatrixXd forward(const Eigen::MatrixXd& in_mat, bool train_flag) override
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

    Eigen::MatrixXd backward(const Eigen::MatrixXd& in_mat) override
    {
        Eigen::MatrixXd out_mat(in_mat.rows(), in_mat.cols());
        for (int i = 0; i < in_mat.rows(); i++) {
            for (int j = 0; j < in_mat.cols(); j++) {
                out_mat(i, j) = (in_mat(i, j) > 0) ? 1. : 0.;
            }
        }
        return out_mat;
    }

private:
};

}  // namespace of MachineLearning
