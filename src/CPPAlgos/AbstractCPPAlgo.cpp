/*! \file AbstractMatrixLoader.h*/
//
// Created by tony on 25/05/23.
//

#include <CPPAlgos/AbstractCPPAlgo.h>
torch::Tensor AMMBench::AbstractCPPAlgo::amm(torch::Tensor A, torch::Tensor B, uint64_t sketchSize) {
  std::cout << sketchSize;
  return torch::matmul(A, B);
}