#pragma once

#include "dnn/abst_layer.hpp"

namespace MachineLearning
{

class Dropout : public AbstLayer
{
public:
    explicit Dropout(double ratio)
        : AbstLayer(), m_ratio(ratio) {}

    void forward() {}
    void backward() {}
private:
    const double m_ratio = 0.0;
};

}  // namespace of MachineLearning
