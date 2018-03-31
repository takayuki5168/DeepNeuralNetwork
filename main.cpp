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
  Eigen::MatrixXd ans_mat(4, 2);
  ans_mat << 1, 0, 0, 1, 0, 1, 1, 0;

  // TODO LSTM Softmax
  auto dnn = initDeepNeuralNetwork();
  dnn->add<Dense>(10, 2);
  dnn->add<ReLU>();
  dnn->add<Dense>(20);
  dnn->add<ReLU>();
  dnn->add<Dense>(10);
  dnn->add<ReLU>();
  dnn->add<Dense>(2);
  //dnn->add<Softmax>();
  //dnn->add<LSTM>(128);

  // TODO MeanAbsoluteError Crossentropy Hinge BinaryCrossentropy CategoricalCrossentropy
  dnn->loss<MeanSquaredError>();
  // TODO Adam RMSprop
  //dnn->opt<SGD>(0.01);
  //dnn->opt<Momentum>();
  //dnn->opt<Adagrad>();
  //dnn->opt<Adam>();
  dnn->opt<RMSprop>();

  dnn->fit(train_mat, ans_mat, 1000);


  std::cout << std::endl;
  Eigen::MatrixXd out_mat = dnn->predict(train_mat);
  std::cout << "==input==" << std::endl;
  std::cout << train_mat << std::endl;
  std::cout << "==output==" << std::endl;
  std::cout << out_mat << std::endl;

  return 0;
}
