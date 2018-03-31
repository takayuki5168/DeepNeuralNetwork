/*!
 * @file    deep_neural_network.hpp
 * @brief   class of DeepNeuralNetwork
 */
#pragma once

#include <memory>
#include <functional>
#include <random>
#include <chrono>
#include "dnn/abst_layer.hpp"
#include "dnn/loss.hpp"
#include "dnn/optimizer.hpp"

//#define DEBUG_MESSAGE

namespace MachineLearning
{

/*!
 * @brief   class of DeepNeuralNetwork
 */
class DeepNeuralNetwork
{
public:
    /*!
     * @brief   constructor
     */
    explicit DeepNeuralNetwork() {}

    /*!
     * @brief   add new layer
     * @param   layer     new layer added
     */
    void add(std::unique_ptr<AbstLayer>&& layer)
    {
        if (not m_layers.empty()) {            // This layer is not input layer
            if (layer->getNeuronNum() == 0) {  // NeuronNum was not settled
                layer->setInNum(m_layers.back()->getNeuronNum());
                layer->setNeuronNum(m_layers.back()->getNeuronNum());
            } else if (layer->getInNum() == 0) {  // NeuronNum was not settled
                layer->setInNum(m_layers.back()->getNeuronNum());
            }
            DYNAMIC_ASSERT(layer->getNeuronNum() > 0, "NeuronNum should be more than zero.");
        } else {                               // This layer is input layer
            if (layer->getNeuronNum() == 0) {  // NeuronNum was not settled
                layer->setInNum(m_layers.back()->getNeuronNum());
                layer->setNeuronNum(m_layers.back()->getNeuronNum());
            }
        }
        layer->initNetwork();
        m_layers.push_back(std::move(layer));

        m_d_loss_func = [](const Eigen::MatrixXd& in_mat, const Eigen::MatrixXd& ans_mat) {
            return (in_mat - ans_mat);
        };
    }

    /*!
     * @brief   fit
     * @param   in_mat     input matrix of train_data
     * @param   ans_mat    answer matrix of train_data
     */
  void fit(const Eigen::MatrixXd& in_mat, const Eigen::MatrixXd& ans_mat, int epoch = 1000, int batch_size = 4)
    {
        setbuf(stdout, NULL);
	Eigen::MatrixXd batch_in_mat(batch_size, in_mat.cols());
	Eigen::MatrixXd batch_ans_mat(batch_size, ans_mat.cols());	
        for (int i = 0; i < epoch; i++) {
	    auto start_time = std::chrono::system_clock::now();
            std::cout << "Epoch " << i + 1 << "/" << epoch << std::endl;
            int loop_num = std::max(static_cast<int>(static_cast<double>(in_mat.rows()) / batch_size), 1);
	    for (int j = 0; j < loop_num; j++){
	        for (int k = 0; k < batch_size; k++) {
                    mt.seed(rnd());
	      	    int row = static_cast<int>(mt()) % static_cast<int>(in_mat.rows());
	      	    row = row > 0 ? row : -row;
	      	    //std::cout << row << std::endl;
		    // TODO TODO TODO
  	            batch_in_mat.row(k) = in_mat.row(k);//row);
	      	    batch_ans_mat.row(k) = ans_mat.row(k);//row);
               }
	        //std::cout << batch_in_mat << std::endl;
	      	//std::cout << batch_ans_mat << std::endl;
	        	    
      	      
               // forward
                Eigen::MatrixXd next_in_mat = batch_in_mat;
                for (unsigned int k = 0; k < m_layers.size(); k++) {
   	          //std::cout << i << " " << next_in_mat << std::endl;
                   Eigen::MatrixXd tmp_mat = m_layers.at(k)->forward(next_in_mat, true);
                   next_in_mat.resize(tmp_mat.rows(), tmp_mat.cols());
                   next_in_mat = tmp_mat;
                }
	        
	        double loss = m_loss_func(next_in_mat, batch_ans_mat);
	        
	        
                // backward
                Eigen::MatrixXd tmp_mat = m_d_loss_func(next_in_mat, batch_ans_mat);
	        
#ifdef DEBUG_MESSAGE	
                std::cout << "[Backward]" << std::endl;
                std::cout << next_in_mat << std::endl;
                std::cout << batch_ans_mat << std::endl;
#endif	        
	        
	        next_in_mat.resize(tmp_mat.rows(), tmp_mat.cols());
                next_in_mat = tmp_mat;
	        
                for (unsigned int k = 0; k < m_layers.size(); k++) {
                    Eigen::MatrixXd tmp_mat = m_layers.at(m_layers.size() - k - 1)->backward(next_in_mat);
	            m_opt_func(m_layers.at(m_layers.size() - k - 1));
	            
                    next_in_mat.resize(tmp_mat.rows(), tmp_mat.cols());
                    next_in_mat = tmp_mat;
               }
	       auto end_time = std::chrono::system_clock::now();
	       double elapsed = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()) / 1000;
	       
	       printf("\r");
	       printf("%d/%d [", j + 1, loop_num);
	       	       int width = 15;
	       for (int k = 0; k < width; k++) {
		 if (k < static_cast<double>(width) * (j + 1) / loop_num) {
		   printf("=");
		 }else{
		   printf(" ");
		 }
	       }
	 
	       printf("] %2.3lfs  loss: %3.3lf", elapsed, loss);
	    }
	    std::cout << std::endl;
	   
        }
    }

    /*!
     * @brief   predict
     * @param   in_mat     input matrix to predict
     */
    Eigen::MatrixXd predict(const Eigen::MatrixXd& in_mat)
    {
        Eigen::MatrixXd next_in_mat = in_mat;
	//std::cout << "[Predict]" << std::endl;
        for (unsigned int i = 0; i < m_layers.size(); i++) {
	  //std::cout << next_in_mat << std::endl;
            Eigen::MatrixXd tmp_mat = m_layers.at(i)->forward(next_in_mat, false);
            next_in_mat.resize(tmp_mat.rows(), tmp_mat.cols());
            next_in_mat = tmp_mat;
        }
        return next_in_mat;
    }

  void compile(std::unique_ptr<AbstLoss> loss, std::unique_ptr<AbstOptimizer> optimizer)
  {
      m_loss_func = loss->getLossFunc();
      m_d_loss_func = loss->getDLossFunc();
      m_opt_func = optimizer->getOptFunc();
  }

private:
    std::vector<std::unique_ptr<AbstLayer>> m_layers;                              //!< layers
    std::function<double(Eigen::MatrixXd, Eigen::MatrixXd)> m_loss_func;  //!< loss function  
    std::function<Eigen::MatrixXd(Eigen::MatrixXd, Eigen::MatrixXd)> m_d_loss_func;  //!< derivation of loss function
  std::function<void(std::unique_ptr<AbstLayer>&)> m_opt_func;

    std::random_device rnd;
    std::mt19937 mt;

};

}  // namespace of MachineLearning
