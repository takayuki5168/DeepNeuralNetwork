#include <gtest/gtest.h>
#include "dnn/activation.hpp"

namespace
{
using namespace MachineLearning;

TEST(SoftmaxTest, Forward)
{
    Eigen::MatrixXd train_data(4, 2);
    train_data << 0, 0, 0, 1, 1, 0, 1, 1;
    train_data = train_data.transpose();

    auto softmax = std::make_unique<Softmax>();
    softmax->forwardWithPredict(train_data);
    EXPECT_EQ(softmax->getOutMat(), train_data);
}

TEST(SoftmaxTest, Backward)
{
    Eigen::MatrixXd train_data(4, 2);
    train_data << 0, 0, 0, 1, 1, 0, 1, 1;
    train_data = train_data.transpose();

    auto softmax = std::make_unique<Softmax>();
    softmax->initNetwork();
    softmax->forwardWithPredict(train_data);
    EXPECT_EQ(softmax->getOutMat(), train_data);
}

}  // anonymous namespace
