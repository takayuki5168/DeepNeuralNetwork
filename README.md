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
std::unique_ptr<DeepNeuralNetwork> dnn = std::make_unique<DeepNeuralNetwork>();

dnn->add(std::make_unique<Dense>(10, 4));
dnn->add(std::make_unique<Relu>());
dnn->add(std::make_unique<Dense>(4));
dnn->add(std::make_unique<Dropout>(0.1));
dnn->add(std::make_unique<Softmax>());

dnn->compile(Crossentropy(), Adam());
```

## Test
---
```sh
$ ./dense_gtest
```
