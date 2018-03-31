#include <iostream>
#include "dnn/deep_neural_network.hpp"
#include "dnn/dense.hpp"
#include "dnn/activation.hpp"
#include "dnn/dropout.hpp"
#include "dnn/loss.hpp"
#include "dnn/optimizer.hpp"

int main()
{
  using namespace MachineLearning;

  Eigen::MatrixXd train_mat(4, 2);
  train_mat << 0, 0, 0, 1, 1, 0, 1, 1;
  
  Eigen::MatrixXd ans_mat(4, 4);
  ans_mat << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;

  auto dnn = initDeepNeuralNetwork();
  dnn->add<Dense>(10, 2);
  dnn->add<ReLU>();
  dnn->add<Dense>(20);
  dnn->add<ReLU>();
  dnn->add<Dense>(10);
  dnn->add<ReLU>();
  dnn->add<Dense>(4);

  dnn->loss<MeanSquaredError>();
  dnn->opt<RMSprop>();
  
  dnn->fit(train_mat, ans_mat, 1000);

  return 0;
}
