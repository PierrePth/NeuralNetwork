#include "neural_net.hh"

NeuralNet::NeuralNet(size_t in, size_t hidden, size_t outs, double learning_rate)
{
  input_ = Tools::create_array(in);
  hidden_ = Tools::create_array(hidden);
  output_ = Tools::create_array(outs);
  w1_ = Tools::create_2D_array(hidden, in);
  w2_ = Tools::create_2D_array(outs, hidden);
  error_out_ = Tools::create_array(outs);
  output_gradient_ = Tools::create_2D_array(outs, hidden);
  hidden_gradient_ = Tools::create_2D_array(hidden, in);
  
  input_nb_ = in;
  hidden_nb_ = hidden;
  output_nb_ = outs;
  
  learning_rate_ = learning_rate;
}

double* NeuralNet::input_get()
{
  return input_;
}

double* NeuralNet::hidden_get()
{
  return hidden_;
}

double* NeuralNet::output_get()
{
  return output_;
}

void NeuralNet::input_set(size_t i, double v)
{
  input_[i] = v;
}

void NeuralNet::hidden_set(size_t i, double v)
{
  hidden_[i] = v;
}

void NeuralNet::output_set(size_t i, double v)
{
  output_[i] = v;
}

void NeuralNet::w1_set(size_t i, size_t j, double v)
{
  w1_[i][j] = v; 
}

void NeuralNet::w2_set(size_t i, size_t j, double v)
{
  w2_[i][j] = v; 
}

double** NeuralNet::w1_get()
{
  return w1_;
}

double** NeuralNet::w2_get()
{
  return w2_;
}

void NeuralNet::output_gradient_set(size_t i, size_t j, double v)
{
  output_gradient_[i][j] = v;
}

void NeuralNet::hidden_gradient_set(size_t i, size_t j, double v)
{
  hidden_gradient_[i][j] = v; 
}

double** NeuralNet::output_gradient_get()
{
  return output_gradient_;
}

double** NeuralNet::hidden_gradient_get()
{
  return hidden_gradient_;
}

double* NeuralNet::error_out_get()
{
  return error_out_;
}

void NeuralNet::error_out_set(size_t i, double v)
{
  error_out_[i] = v;
}

size_t NeuralNet::input_nb_get()
{
  return input_nb_;
}

size_t NeuralNet::hidden_nb_get()
{
  return hidden_nb_;
}

size_t NeuralNet::output_nb_get()
{
  return output_nb_;
}

double NeuralNet::learning_rate_get()
{
  return learning_rate_;
}

void NeuralNet::set_input_layer(const std::vector<double>& v)
{
  for (size_t i = 0; i < v.size(); ++i)
  {
    std::cout << "input : ";
    std::cout << v[i] << " ";
    input_[i] = v[i];
  }
}

void NeuralNet::pretty_print()
{
  std::cout << "===============================" << std::endl;
  std::cout << "inputs : " << std::endl;
  for (size_t i = 0; i < input_nb_; ++i)
    std::cout << input_[i] << std::endl;
  std::cout << "hiddens : " << std::endl;
  for (size_t i = 0; i < hidden_nb_; ++i)
    std::cout << hidden_[i] << std::endl;
  std::cout << "outputs : " << std::endl;
  for (size_t i = 0; i < output_nb_; ++i)
    std::cout << output_[i] << std::endl;
  std::cout << "-----------weights--------" << std::endl;

  for (size_t i = 0; i < hidden_nb_; ++i)
    for (size_t j = 0; j < input_nb_; ++j)
      std::cout << "w1 : " << j << " " << i << ": " << w1_[i][j] << std::endl;

  for (size_t i = 0; i < output_nb_; ++i)
    for (size_t j = 0; j < hidden_nb_; ++j)
      std::cout << "w2 : " << j << " " << i << ": " << w2_[i][j] << std::endl;

}

void NeuralNet::print_res()
{
  std::cout << " output : " << output_[0] << std::endl;
}

std::vector<double> NeuralNet::load_data(std::string file)
{
  std::ifstream f(file);
  std::vector<double> input;
  std::vector<double> output;

  if (!f.is_open())
  {
    std::cerr << "Cannot open file " << file << std::endl;
    throw;
  }
  else
  {
    char c;
    while(f.get(c))
    {
      if (c == '0' || c == '1')
        input.push_back(atoi(&c));
      else if (c == '-')
        break;
    }
    while(f.get(c))
    {
      if (c == '0' || c == '1')
        output.push_back(atoi(&c));
    }
    f.close();
  }
  set_input_layer(input);
  return output;
}
