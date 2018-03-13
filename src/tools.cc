#include "tools.hh"
#include <math.h>

namespace Tools{

  double fRand(double min, double max)
  {
    double f = (double)rand() / RAND_MAX;
    return min + f * (max - min);
  }

  double* create_array(size_t n)
  {
    double *array = new double[n];
    for (size_t i = 0; i < n; ++i)
      array[i] = fRand(0.49, 0.51);
    return array;
  }

  double** create_2D_array(size_t h, size_t w)
  { 
    double **array = new double*[h];
    for (size_t i = 0; i < h; ++i)
    {
      array[i] = new double[w];
      for (size_t j = 0; j < w; ++j)
        array[i][j] = fRand(0.1, 0.99);
    }
    return array;
  }
  double sigmoid(double x)
  {
    return (1.0 / ( 1.0 + exp(-x)));
  }

  std::vector<std::string> list_dir(char *directory)
  {
    DIR *dir;
    struct dirent *ent;
    dir = opendir(directory);
    std::vector<std::string> files;
    if (dir)
    {
      while ((ent = readdir(dir)))
      {
        try
        {
          std::string d(directory);
          std::string f(ent->d_name);
          std::string cur = ".";
          std::string prev = "..";

          if (ent->d_name != cur && ent->d_name != prev)
            files.push_back(d+f);
        }
        catch (std::exception& e)
        {
          throw ("Cannot open file");
        }
      }
      closedir(dir);
    }
    else
    {
      throw std::invalid_argument("Invalid directory");
    }
    return files;
  }
 
}
