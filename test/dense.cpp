#include <gtest/gtest.h>
#include "dnn/dense.hpp"

namespace
{
using namespace MachineLearning;

TEST(DenseTest, Forward)
{
    Eigen::MatrixXd train_data(4, 2);
    train_data << 0, 0, 0, 1, 1, 0, 1, 1;
    train_data = train_data.transpose();

    auto dense = std::make_unique<Dense>(10, 4);
    dense->initNetwork();
    dense->forwardWithPredict(train_data);
    EXPECT_EQ(dense->getOutMat(), train_data);
}

TEST(DenseTest, Backward)
{
    Eigen::MatrixXd train_data(4, 2);
    train_data << 0, 0, 0, 1, 1, 0, 1, 1;
    train_data = train_data.transpose();

    auto dense = std::make_unique<Dense>(10, 4);
    dense->initNetwork();
    dense->forwardWithPredict(train_data);
    EXPECT_EQ(dense->getOutMat(), train_data);
}

}  // anonymous namespace
