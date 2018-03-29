#include <iostream>
#include "dnn/deep_neural_network.hpp"
#include "dnn/dense.hpp"
#include "dnn/dropout.hpp"
#include "dnn/activation.hpp"
#include "dnn/loss.hpp"
#include "dnn/optimizer.hpp"

int main()
{
    using namespace MachineLearning;

    Eigen::MatrixXd train_mat(4, 2);
    train_mat << 0, 0, 0, 1, 1, 0, 1, 1;
    Eigen::MatrixXd ans_mat(4, 1);
    ans_mat << 1, 0, 0, 1;

    std::unique_ptr<DeepNeuralNetwork> dnn = std::make_unique<DeepNeuralNetwork>();
    dnn->add(std::make_unique<Dense>(10, 2));
    dnn->add(std::make_unique<ReLU>());
    dnn->add(std::make_unique<Dense>(20));
    dnn->add(std::make_unique<ReLU>());
    dnn->add(std::make_unique<Dense>(5));
    dnn->add(std::make_unique<ReLU>());
    //dnn->add(std::make_unique<Sigmoid>());    
    dnn->add(std::make_unique<Dense>(1));
    // TODO 出力数が1のものにsoftmaxはつかっては行けない

    //dnn->add(std::make_unique<Dropout>(0.1));
    //dnn->add(std::make_unique<Dense>(10));
    //dnn->add(std::make_unique<Dropout>(0.1));
    //dnn->add(std::make_unique<LSTM>(128));

    dnn->compile(std::make_unique<MeanSquaredError>(), std::make_unique<SGD>());

    for (int i = 0; i < 10000; i++) {
        dnn->fit(train_mat, ans_mat);
    }
    
    Eigen::MatrixXd out_mat = dnn->predict(train_mat);
    std::cout << "==input==" << std::endl;
    std::cout << train_mat << std::endl;
    std::cout << "==output==" << std::endl;
    std::cout << out_mat << std::endl;


    return 0;
}
