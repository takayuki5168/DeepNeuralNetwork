#pragma once

#include "dnn/abst_layer.hpp"
#include "dnn/math_util.hpp"

namespace MachineLearning
{
class Identify : public AbstLayer
{
public:
    explicit Identify() : AbstLayer() {}

    Eigen::VectorXd forward(Eigen::VectorXd in_mat) override
    {
        return in_mat;
    }
    Eigen::VectorXd backward(Eigen::VectorXd in_mat) override
    {
        return in_mat;
    }

private:
};

class Step : public AbstLayer
{
public:
    explicit Step() : AbstLayer() {}

    Eigen::VectorXd forward(Eigen::VectorXd in_mat) override
    {
        Eigen::VectorXd out_mat;
        for (int i = 0; i < in_mat.size(); i++) {
            out_mat(i) = (in_mat(i) > 0) ? 1 : 0;
        }
        return out_mat;
    }
    // TODO
    Eigen::VectorXd backward(Eigen::VectorXd in_mat) override
    {
        return in_mat;
    }

private:
};

class Softmax : public AbstLayer
{
public:
    explicit Softmax() : AbstLayer() {}

    Eigen::VectorXd forward(Eigen::VectorXd in_mat) override
    {
        // TODO 最小値でいいのか
        double min_mat = in_mat(0);
        for (int i = 0; i < in_mat.size(); i++) {
            min_mat = (min_mat > in_mat(i)) ? in_mat(i) : min_mat;
        }

        Eigen::VectorXd exp_mat;
        exp_mat.resize(in_mat.size());
        for (int i = 0; i < in_mat.size(); i++) {
            exp_mat(i) = std::exp(in_mat(i) - min_mat);
        }
        return exp_mat;
    }
    // TODO
    Eigen::VectorXd backward(Eigen::VectorXd in_mat) override
    {
        return in_mat;
    }

private:
};

class Sigmoid : public AbstLayer
{
public:
    explicit Sigmoid() : AbstLayer() {}

    Eigen::VectorXd forward(Eigen::VectorXd in_mat) override
    {
        Eigen::VectorXd out_mat;
        for (int i = 0; i < in_mat.size(); i++) {
            out_mat(i) = 1.0 + (1.0 + exp(-in_mat(i)));
        }
        return out_mat;
    }
    Eigen::VectorXd backward(Eigen::VectorXd in_mat) override
    {
        return in_mat;
    }

private:
};

class Relu : public AbstLayer
{
public:
    explicit Relu() : AbstLayer() {}

    Eigen::VectorXd forward(Eigen::VectorXd in_mat) override
    {
        Eigen::VectorXd out_mat;
        for (int i = 0; i < in_mat.size(); i++) {
            out_mat(i) = (in_mat(i) > 0) ? in_mat(i) : 0;
        }
        return out_mat;
    }
    Eigen::VectorXd backward(Eigen::VectorXd in_mat) override
    {
        return in_mat;
    }

private:
};

}  // namespace of MachineLearning
