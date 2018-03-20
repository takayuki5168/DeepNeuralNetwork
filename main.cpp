#include <iostream>
#include "dnn/deep_neural_network.hpp"
#include "dnn/dense.hpp"
#include "dnn/dropout.hpp"
#include "dnn/activation.hpp"

int main()
{
    using namespace MachineLearning;

    Eigen::MatrixXd train_data(4, 2);
    train_data << 0, 0, 0, 1, 1, 0, 1, 1;
    train_data = train_data.transpose();

    Eigen::MatrixXd answer_data(4, 1);
    answer_data << 1, 0, 0, 1;
    answer_data = answer_data.transpose();

    std::unique_ptr<DeepNeuralNetwork> dnn = std::make_unique<DeepNeuralNetwork>();
    dnn->add(std::make_unique<Dense>(4, 4));
    dnn->add(std::make_unique<Dense>(10));
    dnn->add(std::make_unique<Dense>(4));
    //dnn->add(std::make_unique<Softmax>());
    //dnn->add(std::make_unique<Dropout>(0.1));
    //dnn->add(std::make_unique<Dense>(10));
    //dnn->add(std::make_unique<Relu>());
    //dnn->add(std::make_unique<Dropout>(0.1));
    //dnn->add(std::make_unique<LSTM>(128));

    //dnn->compile(Crossentropy(), Adam());

    //dnn->fit(train_data);
    Eigen::MatrixXd out_mat = dnn->predict(train_data);
    std::cout << out_mat << std::endl;


    return 0;
}
