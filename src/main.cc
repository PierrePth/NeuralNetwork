#include <iostream>
#include <stdlib.h>
#include <vector>
#include "tools.hh"
#include "neural_net.hh"

/** \name Feed forward propagation function

    \brief This function updates the neurons by calculating
    the sigmoid of the logit

    \param The neural network
*/
void feed_forward(NeuralNet& nn)
{
  for (size_t i = 0; i < nn.hidden_nb_get(); ++i)
  {
    double sum = 0.0;
    for (size_t j = 0; j < nn.input_nb_get(); ++j)
    {
      sum += nn.input_get()[j] * nn.w1_get()[i][j];
    }
    nn.hidden_set(i, Tools::sigmoid(sum));
  }
  
  for (size_t i = 0; i < nn.output_nb_get(); ++i)
  {
    double sum = 0.0;
    for (size_t j = 0; j < nn.hidden_nb_get(); ++j)
    {
      sum += nn.hidden_get()[j] * nn.w2_get()[i][j];
    }

    nn.output_set(i, Tools::sigmoid(sum));
  } 
}

/** \name Trains the Neural network

    \brief This function first calcul the output error.
    Then, it computes the gradients of the weights that
    goes from the hidden layer to the output layer by
    applying the formula of the partial derivative of
    the error function E thanks to the weights W.
    It does the same thing with the first matrix
    of weights, this time adding adding the sum of product
    of each weight from the w2 matrix in order to retropropagate
    the error from the output correctly. Finally, we update the
    weights of the 2 matrix simply by multiplying the gradient by
    the learning rate.

    \param The neural network and the expected output
*/
void train(NeuralNet& nn, std::vector<double> out_data)
{
  for (size_t i = 0; i < out_data.size(); ++i)
    nn.error_out_set(i, out_data[i] - nn.output_get()[i]);

  for (size_t i = 0; i < out_data.size(); ++i)
  {
    for (size_t j = 0; j < nn.hidden_nb_get(); ++j)
    {
      double grad = (-nn.error_out_get()[i]) * nn.output_get()[i] *
                    (1 - nn.output_get()[i]) * nn.hidden_get()[j];
      nn.output_gradient_set(i, j, grad);
    }
  }

  for (size_t i = 0; i < nn.hidden_nb_get(); ++i)
  {
    for (size_t j = 0; j < nn.input_nb_get(); ++j)
    {
      double sum = 0.0;
      for (size_t x = 0; x < nn.output_nb_get(); ++x)
        sum += nn.w1_get()[x][i] * nn.error_out_get()[x];
      double grad = (-sum) * nn.input_get()[j] *
                    (1 - nn.hidden_get()[i]) * nn.hidden_get()[i];
      nn.hidden_gradient_set(i, j, grad);
    }
  }
  
  for (size_t i = 0; i < nn.output_nb_get(); ++i)
  {
    for (size_t j = 0; j < nn.hidden_nb_get(); ++j)
    {
      double old_weight = nn.w2_get()[i][j];
      nn.w2_set(i, j, old_weight - nn.learning_rate_get()
                * nn.output_gradient_get()[i][j]);
    }
  }

  for (size_t i = 0; i < nn.hidden_nb_get(); ++i)
  {
    for (size_t j = 0; j < nn.input_nb_get(); ++j)
    {
      double old_weight = nn.w1_get()[i][j];
      nn.w1_set(i, j, old_weight - nn.learning_rate_get()
                * nn.hidden_gradient_get()[i][j]);
    }
  }
}

/** \name main function

    \brief This function parse the data_sets files
    and run a SGD which trains the network the files.

    \param Usual parameters
*/
int main(int argc, char **argv)
{
  if (argc < 7)
  {
    std::cerr << "Usage : ./nn <data_set> <inputs> <hiddens> <outputs>";
    std::cerr << " <learning_rate> <number of epochs>" << std::endl;
    return 1;
  }
  std::vector<std::string> files = Tools::list_dir(argv[1]);
  
  NeuralNet nn(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atof(argv[5])); 

  std::vector<double> exp_out = nn.load_data(files[0]);

  nn.pretty_print();

  //  case of stochastic gradient descent as example
  //  we have 50000 / number of files in data_set epochs
  int epochs = files.size() * atoi(argv[6]);
  for (int i = 0; i < epochs; ++i)
  {
    int r_index = Tools::fRand(0, files.size());
    exp_out = nn.load_data(files[r_index]);
    feed_forward(nn);
    train(nn, exp_out);
    nn.print_res();
  }

  return 0;
}
