/*
 * @file    loss.hpp
 * @brief   classes of Loss Function
 */
#pragma once

#include "dnn/math_util.hpp"

namespace MachineLearning
{

class AbstLoss
{
public:
    explicit AbstLoss(){}

    virtual std::function<double(Eigen::MatrixXd, Eigen::MatrixXd)> getLossFunc() const { return m_loss_func; }
    virtual std::function<Eigen::MatrixXd(Eigen::MatrixXd, Eigen::MatrixXd)> getDLossFunc() const { return m_d_loss_func; }
  
protected:
    std::function<double(Eigen::MatrixXd, Eigen::MatrixXd)> m_loss_func;
    std::function<Eigen::MatrixXd(Eigen::MatrixXd, Eigen::MatrixXd)> m_d_loss_func;  
};

class MeanSquaredError : public AbstLoss
{
public:
    explicit MeanSquaredError ()
      : AbstLoss()
    {
      //TODO
        m_d_loss_func = [](Eigen::MatrixXd in_mat, Eigen::MatrixXd ans_mat) { return in_mat - ans_mat; };
        m_loss_func = [](Eigen::MatrixXd in_mat, Eigen::MatrixXd ans_mat)
	{
	    double error = 0.;
	    for (int i = 0; i < in_mat.rows(); i++) {
 	        for (int j = 0; j < in_mat.cols(); j++) {
		  error += std::pow(in_mat(i, j) - ans_mat(i, j), 2);
		}
	    }
	    return error;
	};	
    }
};

  //TODO
class MeanAbsoluteError : public AbstLoss
{
public:
    explicit MeanAbsoluteError ()
      : AbstLoss()
    {
        m_d_loss_func = [](Eigen::MatrixXd in_mat, Eigen::MatrixXd ans_mat) { return in_mat - ans_mat; };
        m_loss_func = [](Eigen::MatrixXd in_mat, Eigen::MatrixXd ans_mat)
	{
	    double error = 0.;
	    for (int i = 0; i < in_mat.rows(); i++) {
 	        for (int j = 0; j < in_mat.cols(); j++) {
		  error += std::pow(in_mat(i, j) - ans_mat(i, j), 2);
		}
	    }
	    return error;
	};	
    }
};

  //TODO
class CrossEntropy : public AbstLoss
{
public:
    explicit CrossEntropy ()
      : AbstLoss()
    {
        m_d_loss_func = [](Eigen::MatrixXd in_mat, Eigen::MatrixXd ans_mat) { return in_mat - ans_mat; };
        m_loss_func = [](Eigen::MatrixXd in_mat, Eigen::MatrixXd ans_mat)
	{
	    double error = 0.;
	    for (int i = 0; i < in_mat.rows(); i++) {
 	        for (int j = 0; j < in_mat.cols(); j++) {
		  error += std::pow(in_mat(i, j) - ans_mat(i, j), 2);
		}
	    }
	    return error;
	};	
    }
};

}  // namespace of MachineLearning
