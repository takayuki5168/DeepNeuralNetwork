/*
 *
 */
#pragma once

namespace MachineLearning
{

class AbstLayer
{
public:
    explicit AbstLayer() {}
    virtual void forward() {}
    virtual void backward() {}
private:
};

}  // namespace of MachineLearning
