#pragma once

#include <cstddef>
#include <stdlib.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <dirent.h>

namespace Tools
{
  double fRand(double min, double max);
  double* create_array(size_t n);
  double** create_2D_array(size_t h, size_t w);
  double sigmoid(double x);
  std::vector<std::string> list_dir(char *directory);
}
