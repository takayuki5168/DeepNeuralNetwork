/*
 *
 */
#pragma once

#include <memory>

namespace MachineLearning
{

class AbstLayer
{
public:
    explicit AbstLayer() {}
    //AbstLayer(std::unique_ptr<AbstLayer>&& abst_layer)
    //    : {}
    virtual void forward() {}
    virtual void backward() {}
private:
    std::vector<std::vector<float>> m_weights;
};

}  // namespace of MachineLearning
