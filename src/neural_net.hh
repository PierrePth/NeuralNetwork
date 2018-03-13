#pragma once

#include "tools.hh"

class NeuralNet
{
  public:
    NeuralNet(size_t in, size_t hidden, size_t outs, double learning_rate);
    double*  input_get();
    double*  hidden_get();
    double*  output_get();
    void  input_set(size_t i, double v);
    void  hidden_set(size_t i, double v);
    void  output_set(size_t i, double v);
    void  w1_set(size_t i, size_t j, double v);
    void  w2_set(size_t i, size_t j, double v);
    double**  w1_get();
    double**  w2_get();
    void  output_gradient_set(size_t i, size_t j, double v);
    void  hidden_gradient_set(size_t i, size_t j, double v);
    double**  output_gradient_get();
    double**  hidden_gradient_get();
    double*  error_out_get();
    void  error_out_set(size_t i, double v);
    size_t  input_nb_get();
    size_t  hidden_nb_get();
    size_t  output_nb_get();
    double  learning_rate_get();
    void set_input_layer(const std::vector<double>& v);
    void pretty_print();
    void print_res();
    std::vector<double> load_data(std::string s);
  private: 
    double* input_;
    double* hidden_;
    double* output_;
    double** w1_;
    double** w2_;

    double *error_out_;
    double **output_gradient_;
    double **hidden_gradient_;

    size_t input_nb_;
    size_t hidden_nb_;
    size_t output_nb_;

    double learning_rate_;
};
