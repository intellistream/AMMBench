//
// Created by haolan on 6/5/23.
//
#include <MatrixLoader/MediaMillMatrixLoader.h>
#include <Utils/IntelliLog.h>
#include <AMMBench.h>
#include <fstream>
#include <iostream>

void AMMBench::MediaMillMatrixLoader::paraseConfig(INTELLI::ConfigMapPtr cfg) {
  filePath = cfg->tryString("filePath", "datasets/Mediamill/mediamill.pth", true);
}

void AMMBench::MediaMillMatrixLoader::generateAB() {

  torch::jit::script::Module tensors = torch::jit::load(filePath);
  // A, B already normalized
  A = tensors.attr("A").toTensor().contiguous(); // 120*43907
  B = tensors.attr("B").toTensor().contiguous(); // 101*43907

  At = A.t().contiguous(); // 43907*120
  Bt = B.t().contiguous(); // 43907*101

  std::cout << "Maximum Value: " << A.max().item<float>() << std::endl;
  std::cout << "Mean Value: " << A.mean().item<float>() << std::endl;
  std::cout << "Minimum Value: " << A.min().item<float>() << std::endl;
  std::cout << "Maximum Value: " << B.max().item<float>() << std::endl;
  std::cout << "Mean Value: " << B.mean().item<float>() << std::endl;
  std::cout << "Minimum Value: " << B.min().item<float>() << std::endl;

  INTELLI_INFO(
      "Generating [" + to_string(A.size(0)) + " x " + to_string(A.size(1)) + "]*[" + to_string(B.size(0)) + " x "
          + to_string(B.size(1)) + "]");
}

//do nothing in abstract class
bool AMMBench::MediaMillMatrixLoader::setConfig(INTELLI::ConfigMapPtr cfg) {
  paraseConfig(cfg);
  generateAB();
  return true;
}

torch::Tensor AMMBench::MediaMillMatrixLoader::getA() {
  return A;
}

torch::Tensor AMMBench::MediaMillMatrixLoader::getB() {
  return B;
}

torch::Tensor AMMBench::MediaMillMatrixLoader::getAt() {
  return At;
}

torch::Tensor AMMBench::MediaMillMatrixLoader::getBt() {
  return Bt;
}

torch::Tensor AMMBench::MediaMillMatrixLoader::getSxx() {
  return Sxx;
}

torch::Tensor AMMBench::MediaMillMatrixLoader::getSyy() {
  return Syy;
}

torch::Tensor AMMBench::MediaMillMatrixLoader::getSxy() {
  return Sxy;
}

torch::Tensor AMMBench::MediaMillMatrixLoader::getSxxNegativeHalf() {
  return SxxNegativeHalf;
}

torch::Tensor AMMBench::MediaMillMatrixLoader::getSyyNegativeHalf() {
  return SyyNegativeHalf;
}

torch::Tensor AMMBench::MediaMillMatrixLoader::getM() {
  return M;
}

torch::Tensor AMMBench::MediaMillMatrixLoader::getM1() {
  return M1;
}

torch::Tensor AMMBench::MediaMillMatrixLoader::getCorrelation() {
  return correlation;
}

void AMMBench::MediaMillMatrixLoader::calculate_correlation() {
  
  // Sxx, Syy, Sxy: covariance matrix
  Sxx = torch::matmul(A, At) / A.size(1); // 120*120
  Syy = torch::matmul(B, Bt) / A.size(1); // 101*101
  Sxy = torch::matmul(A, Bt) / A.size(1); // 120*101

  // Sxx^(-1/2), Syy^(-1/2), M
  // Sxx^(-1/2) 120*120
  torch::Tensor eigenvaluesSxx, eigenvectorsSxx;
  std::tie(eigenvaluesSxx, eigenvectorsSxx) = torch::linalg::eig(Sxx); // diagonization
  torch::Tensor diagonalMatrixSxx = torch::diag(
      1.0 / torch::sqrt(eigenvaluesSxx + torch::full({}, 1e-12))); // 1/sqrt(eigenvalue+epsilon) +epsilon to avoid nan
  SxxNegativeHalf = torch::matmul(torch::matmul(eigenvectorsSxx, diagonalMatrixSxx), eigenvectorsSxx.t());
  SxxNegativeHalf = at::real(SxxNegativeHalf); // ignore complex part, it comes from numerical computations
  // Syy^(-1/2) 101*101
  torch::Tensor eigenvaluesSyy, eigenvectorsSyy;
  std::tie(eigenvaluesSyy, eigenvectorsSyy) = torch::linalg::eig(Syy);
  torch::Tensor diagonalMatrixSyy = torch::diag(1.0 / torch::sqrt(eigenvaluesSyy + torch::full({}, 1e-12)));
  SyyNegativeHalf = torch::matmul(torch::matmul(eigenvectorsSyy, diagonalMatrixSyy), eigenvectorsSyy.t());
  SyyNegativeHalf = at::real(SyyNegativeHalf);
  // M 120*101
  M1 = torch::matmul(SxxNegativeHalf.t(), Sxy);
  M = torch::matmul(M1, SyyNegativeHalf);

  // correlation
  torch::Tensor U, S, Vh;
  std::tie(U, S, Vh) = torch::linalg::svd(M, false, c10::nullopt);
  correlation = torch::clamp(S, -1.0, 1.0);
}
