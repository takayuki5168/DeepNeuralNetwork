#pragma once

#include "dnn/abst_layer.hpp"
#include "dnn/math_util.hpp"

namespace MachineLearning
{
class Identify : public AbstLayer
{
public:
    explicit Identify() : AbstLayer() {}

    Eigen::VectorXd forward(Eigen::VectorXd in_val)
    {
        return in_val;
    }
    Eigen::VectorXd backward(Eigen::VectorXd in_val)
    {
        return in_val;
    }

private:
};

class Step : public AbstLayer
{
public:
    explicit Step() : AbstLayer() {}

    Eigen::VectorXd forward(Eigen::VectorXd in_val)
    {
        Eigen::VectorXd out_val;
        for (int i = 0; i < in_val.size(); i++) {
            out_val(i) = (in_val(i) > 0) ? 1 : 0;
        }
        return out_val;
    }
    // TODO
    Eigen::VectorXd backward(Eigen::VectorXd in_val)
    {
        return in_val;
    }

private:
};

class Softmax : public AbstLayer
{
public:
    explicit Softmax() : AbstLayer() {}

    Eigen::VectorXd forward(Eigen::VectorXd in_val)
    {
        // TODO 最小値でいいのか
        double min_val = in_val(0);
        for (int i = 0; i < in_val.size(); i++) {
            min_val = (min_val > in_val(i)) ? in_val(i) : min_val;
        }

        Eigen::VectorXd exp_val;
        exp_val.resize(in_val.size());
        for (int i = 0; i < in_val.size(); i++) {
            exp_val(i) = std::exp(in_val(i) - min_val);
        }
        return exp_val;
    }
    // TODO
    Eigen::VectorXd backward(Eigen::VectorXd in_val)
    {
        return in_val;
    }

private:
};

class Sigmoid : public AbstLayer
{
public:
    explicit Sigmoid() : AbstLayer() {}

    Eigen::VectorXd forward(Eigen::VectorXd in_val)
    {
        Eigen::VectorXd out_val;
        for (int i = 0; i < in_val.size(); i++) {
            out_val(i) = 1.0 + (1.0 + exp(-in_val(i)));
        }
        return out_val;
    }
    Eigen::VectorXd backward(Eigen::VectorXd in_val)
    {
        return in_val;
    }

private:
};

class Relu : public AbstLayer
{
public:
    explicit Relu() : AbstLayer() {}

    Eigen::VectorXd forward(Eigen::VectorXd in_val)
    {
        Eigen::VectorXd out_val;
        for (int i = 0; i < in_val.size(); i++) {
            out_val(i) = (in_val(i) > 0) ? in_val(i) : 0;
        }
        return out_val;
    }
    Eigen::VectorXd backward(Eigen::VectorXd in_val)
    {
        return in_val;
    }

private:
};

}  // namespace of MachineLearning
