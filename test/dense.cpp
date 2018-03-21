#include <gtest/gtest.h>
#include "dnn/dense.hpp"

namespace
{
using namespace MachineLearning;

TEST(DenseTest, Forward)
{
    Eigen::MatrixXd train_data(4, 2);
    train_data << 0, 0, 0, 1, 1, 0, 1, 1;
    train_data = train_data.transpose();  //!< in_num * train_data_num

    Eigen::MatrixXd answer_data(10, 4);
    answer_data << 2, 1, 1, 1, 3, 2, 1, 1, 2, 1, 1, 1, 3, 1, 1, 1, 2, 1, 1, 1, 3, 1, 1, 1, 2, 1, 1, 1, 3, 1, 1, 1, 2, 1, 1, 1, 3, 1, 1, 1;

    std::unique_ptr<Dense> dense = std::make_unique<Dense>(10, 2);
    dense->initNetwork();
    dense->forwardWithPredict(train_data);

    EXPECT_EQ(dense->getOutMat(), answer_data);
}

/*
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
*/

}  // anonymous namespace
