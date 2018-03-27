#include <iostream>
#include "dnn/deep_neural_network.hpp"
#include "dnn/dense.hpp"
#include "dnn/dropout.hpp"
#include "dnn/activation.hpp"

int main()
{
    using namespace MachineLearning;

    Eigen::MatrixXd train_mat(4, 2);
    train_mat << 0, 0, 0, 1, 1, 0, 1, 1;

    Eigen::MatrixXd ans_mat(4, 1);
    ans_mat << 1, 0, 0, 0;

    std::unique_ptr<DeepNeuralNetwork> dnn = std::make_unique<DeepNeuralNetwork>();
    dnn->add(std::make_unique<Dense>(3, 2));
    //dnn->add(std::make_unique<Relu>());
    dnn->add(std::make_unique<Dense>(10));
    //dnn->add(std::make_unique<Relu>());
    dnn->add(std::make_unique<Dense>(1));
    // TODO 出力数が1のものにsoftmaxはつかっては行けない
    //dnn->add(std::make_unique<Softmax>());

    //dnn->add(std::make_unique<Dropout>(0.1));
    //dnn->add(std::make_unique<Dense>(10));
    //dnn->add(std::make_unique<Dropout>(0.1));
    //dnn->add(std::make_unique<LSTM>(128));

    //dnn->compile(Crossentropy(), Adam());

    for (int i = 0; i < 1000; i++) {
        dnn->fit(train_mat, ans_mat);
    }
    Eigen::MatrixXd out_mat = dnn->predict(train_mat);
    std::cout << "==input==" << std::endl;
    std::cout << train_mat << std::endl;
    std::cout << "==output==" << std::endl;
    std::cout << out_mat << std::endl;


    return 0;
}
