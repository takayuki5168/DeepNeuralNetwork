# DeepNeuralNetwork Library for C++
Implementaion is influenced by [Keras](https://keras.io/)
---

## How to run a sample program
---
```sh
$ mkdir build
$ cd build
$ cmake ..
$ make
$ ./main
```

## Tutorial
### MLP
---
```cpp
// init network
auto dnn = initDeepNeuralNetwork();

// build model
dnn->add<Dense>(10, 2);
dnn->add<ReLU>();
dnn->add<Dense>(20);
dnn->add<ReLU>();
dnn->add<Dense>(10);
dnn->add<ReLU>();
dnn->add<Dense>(4);

// set loss and optimizer
dnn->loss<MeanSquaredError>();
dnn->opt<RMSprop>();

// fit
dnn->fit(train_mat, ans_mat, 1000);
```

### CNN

## Supported networks
- Layer
  - Core
    - Dense
    - Dropout
  - Convolution
  - Normalization
- Activation
  - Sigmoid
  - ReLU
  - Softmax
  - tanh (not implemented)
  - Identity (not implemented)
- Loss
  - MeanSquaredError
  - MeanAbsoluteError
  - CrossEntropy (not implemented)
- Optimizer
  - SGD
  - MomentumSGD
  - AdaGrad
  - RMSprop
  - AdaDelta
  - Adam

## Unit Test
---
```sh
$ ./dense_gtest
```
