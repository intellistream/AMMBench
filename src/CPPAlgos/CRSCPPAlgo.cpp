//
// Created by tony on 25/05/23.
//

#include <CPPAlgos/CRSCPPAlgo.h>

namespace AMMBench {
torch::Tensor AMMBench::CRSCPPAlgo::amm(torch::Tensor A, torch::Tensor B, uint64_t k) {
  A = A.t();
  //INTELLI_INFO("I am CPP-CRS");
  int64_t n = A.size(0);
  //int64_t m = A.size(1);

  assert(n == B.size(0));
  // Probability distribution
  torch::Tensor probs = torch::ones(n) / n;  // default: uniform

  // Sample k indices from range 0 to n for given probability distribution
  torch::Tensor indices = torch::multinomial(probs, k, true);

  // Sample k columns from A
  torch::Tensor A_sampled = A.index_select(0, indices);
  int64_t ratio = std::ceil(static_cast<double>(n) / k);
  A_sampled = (A_sampled / (int) k).t().div(probs.index_select(0, torch::arange(0, n, ratio)));

  // Sample k rows from B
  torch::Tensor B_sampled = B.index_select(0, indices);

  // Compute the matrix product
  torch::Tensor result = torch::matmul(A_sampled, B_sampled);
  return result;
}
} // AMMBench