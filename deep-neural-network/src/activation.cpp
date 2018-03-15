#include "dnn/activation.hpp"

namespace MachineLearning
{

Eigen::VectorXf identify(Eigen::VectorXf in_val) const
{
    Eigen::VectorXf out_val;
    out_val.resize(in_val.size());
    for (int i = 0; i < in_val.size(); i++) {
        out_val[i] = MathUtil::identify(in_val[i]);
    }
    return out_val;
}

Eigen::VectorXf step(Eigen::VectorXf in_val) const
{
    Eigen::VectorXf out_val;
    out_val.resize(in_val.size());
    for (int i = 0; i < in_val.size(); i++) {
        out_val[i] = MathUtil::step(in_val[i]);
    }
    return out_val;
}

Eigen::VectorXf sigmoid(Eigen::VectorXf in_val) const
{
    Eigen::VectorXf out_val;
    out_val.resize(in_val.size());
    for (int i = 0; i < in_val.size(); i++) {
        out_val[i] = MathUtil::sigmoid(in_val[i]);
    }
    return out_val;
}

Eigen::VectorXf relu(Eigen::VectorXf in_val)
{
    Eigen::VectorXf out_val;
    out_val.resize(in_val.size());
    for (int i = 0; i < in_val.size(); i++) {
        out_val[i] = MathUtil::relu(in_val[i]);
    }
    return out_val;
}

Eigen::VectorXf softmax(Eigen::VectorXf in_val)
{
}

}  // namespace of MachineLearning
