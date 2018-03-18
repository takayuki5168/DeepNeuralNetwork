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

    Eigen::VectorXd forward(Eigen::VectorXd in_vec) override { return in_vec; }
    Eigen::VectorXd backward(Eigen::VectorXd in_vec) override { return in_vec; }

private:
};

class Step : public AbstLayer
{
public:
    explicit Step() : AbstLayer() {}

    Eigen::VectorXd forward(Eigen::VectorXd in_vec) override
    {
        Eigen::VectorXd out_vec;
        for (int i = 0; i < in_vec.size(); i++) {
            out_vec(i) = (in_vec(i) > 0) ? 1 : 0;
        }
        return out_vec;
    }
    // TODO
    Eigen::VectorXd backward(Eigen::VectorXd in_vec) override
    {
        return in_vec;
    }

private:
};
*/

class Softmax : public AbstLayer
{
public:
    explicit Softmax() : AbstLayer() {}

private:
    Eigen::VectorXd forward(Eigen::VectorXd in_vec) override
    {
        double max_vec = in_vec(0);
        for (int i = 0; i < in_vec.size(); i++) {
            max_vec = (max_vec > in_vec(i)) ? in_vec(i) : max_vec;
        }

        Eigen::VectorXd exp_vec;
        exp_vec.resize(in_vec.size());
        for (int i = 0; i < in_vec.size(); i++) {
            exp_vec(i) = std::exp(in_vec(i) - max_vec);
        }

        double sum_exp = 0.0;
        for (int i = 0; i < exp_vec.size(); i++) {
            sum_exp += exp_vec(i);
        }
        return exp_vec / sum_exp;
    }
    // TODO
    Eigen::VectorXd backward(Eigen::VectorXd in_vec) override
    {
        return in_vec;
    }
};

class Sigmoid : public AbstLayer
{
public:
    explicit Sigmoid() : AbstLayer() {}

private:
    Eigen::VectorXd forward(Eigen::VectorXd in_vec) override
    {
        for (int i = 0; i < in_vec.size(); i++) {
            m_out_vec(i) = 1.0 / (1.0 + std::exp(-in_vec(i)));
        }
        return m_out_vec;
    }
    Eigen::VectorXd backward(Eigen::VectorXd in_vec) override
    {
        for (int i = 0; i < in_vec.size(); i++) {
            in_vec(i) = in_vec(i) * (1.0 - m_out_vec(i)) * m_out_vec(i);
        }
        return in_vec;
    }
};

class Relu : public AbstLayer
{
public:
    explicit Relu() : AbstLayer() {}

private:
    Eigen::VectorXd forward(Eigen::VectorXd in_vec) override
    {
        for (int i = 0; i < in_vec.size(); i++) {
            m_out_vec(i) = (in_vec(i) > 0) ? in_vec(i) : 0;
        }
        return m_out_vec;
    }
    Eigen::VectorXd backward(Eigen::VectorXd in_vec) override
    {
        for (int i = 0; i < in_vec.size(); i++) {
            in_vec(i) = (in_vec(i) > 0) ? in_vec(i) : 0;
        }
        return in_vec;
    }
};

}  // namespace of MachineLearning
