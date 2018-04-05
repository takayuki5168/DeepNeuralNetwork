# DeepNeuralNetwork Library for C++
---

## How to use
---
```sh
$ mkdir build
$ cd build
$ cmake ..
$ make
$ ./main
```

## Make original model
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

## Unit Test
---
```sh
$ ./dense_gtest
```
