//
// Created by tony on 24/05/23.
//

#include <Parallelization/BlockPartitionRunner.h>
#include <Utils/UtilityFunctions.h>
void AMMBench::BlockPartitionWorker::setConfig(INTELLI::ConfigMapPtr _cfg) {
  cfg = _cfg;
  sketchDimension = cfg->tryU64("sketchDimension", 50, true);
  std::string ptFile = cfg->tryString("ptFile", "torchscripts/FDAMM.pt", true);
  useCPP = cfg->tryU64("useCPP", 0, true);
  if (useCPP) {
    std::string cppAlgoTag = cfg->tryString("cppAlgoTag", "mm", true);
    cppAlgoPtr = cppAlgoTable.findCppAlgo(cppAlgoTag);
  }
  if (!useCPP || (cppAlgoPtr == nullptr)) {
    module = torch::jit::load(ptFile);
    if (useCPP) {
      INTELLI_ERROR("No cpp algorithm found, go back to pt module");
      useCPP = 0;
    }
  }

}
void AMMBench::BlockPartitionWorker::setWorkParameters(uint64_t aStart, uint64_t aEnd, int mycore) {
  startRow = aStart;
  endRow = aEnd;
  coreBind = mycore;
  //matC=newTensor(torch::zeros({(long)(endRow+1-startRow),matB->size(1)}));


}
void AMMBench::BlockPartitionWorker::setABC(AMMBench::TensorPtr A, AMMBench::TensorPtr B, AMMBench::TensorPtr C) {
  matA = A;
  matB = B;
  matC = C;
  //assert(C);
}
void AMMBench::BlockPartitionWorker::inlineMain() {
  //std::cout<<"thread at "+ to_string(coreBind)+" start\r\n";
  /**
   * @brief 1. bind core and torch setting
   */
  // Perform matrix multiplication for the assigned rows
  INTELLI::UtilityFunctions::bind2Core((int) coreBind);
  torch::set_num_threads(1);
  /**
   * @brief 2. multiply sub-matrix of A
   */
  gettimeofday(&tstart, NULL);
  //torch::Tensor
  //torch::Tensor subC =  module.forward({subA, *matB, (long) sketchDimension}).toTensor();
  // Copy the results back to the output matrix C
  //matC->slice(0, startRow, endRow) = subC;
  subA = matA->slice(0, startRow, endRow);
  //irC=module.forward({subA, *matB, (long) sketchDimension}).toTensor();
  if (useCPP) {
    INTELLI_WARNING("USE CPP ALGO");
    matC->slice(0, startRow, endRow) = cppAlgoPtr->amm(subA, *matB, sketchDimension);
  } else {
    matC->slice(0, startRow, endRow) =module.forward({subA, *matB, (long) sketchDimension}).toTensor();
  }

  gettimeofday(&tend, NULL);
  //std::cout<<subC;
}
uint64_t AMMBench::BlockPartitionWorker::getElapsedTime() {
  return INTELLI::UtilityFunctions::timeLast(tstart, tend);
}
void AMMBench::BlockPartitionRunner::setConfig(INTELLI::ConfigMapPtr _cfg) {
  cfg = _cfg;
  threads = cfg->tryU64("threads", 2, true);
  workers = std::vector<BlockPartitionWorkerPtr>(threads);
  for (uint64_t i = 0; i < threads; i++) {
    workers[i] = newBlockPartitionWorker();
    workers[i]->setConfig(cfg);
  }
  INTELLI_INFO("set up " + to_string(threads) + "workers.");
}
void AMMBench::BlockPartitionRunner::createABC(torch::Tensor A, torch::Tensor B) {

  matA = newTensor(A);
  matB = newTensor(B);
  matC = newTensor(torch::zeros({A.size(0), B.size(1)}));
  for (uint64_t i = 0; i < threads; i++) {
    uint64_t rows_per_worker = A.size(0) / threads;
    uint64_t start_row = i * rows_per_worker;
    uint64_t end_row = (i == threads - 1) ? A.size(0) : start_row + rows_per_worker;
    workers[i]->setABC(matA, matB, matC);
    workers[i]->setWorkParameters(start_row, end_row, i);
  }
}
torch::Tensor AMMBench::BlockPartitionRunner::parallelForward() {
  for (uint64_t i = 0; i < threads; i++) {
    workers[i]->startThread();
  }
  INTELLI_INFO("start " + to_string(threads) + "workers.");
  for (uint64_t i = 0; i < threads; i++) {
    workers[i]->joinThread();
  }
  /* for(uint64_t i=0;i<threads;i++)
   {
     matC->slice(0, workers[i]->startRow, workers[i]->endRow) = workers[i]->irC;
   }*/
  return *matC;
}
torch::Tensor AMMBench::BlockPartitionRunner::runAMM(torch::Tensor A, torch::Tensor B) {
  createABC(A, B);
  return parallelForward();
}
uint64_t AMMBench::BlockPartitionRunner::getElapsedTime() {
  uint64_t ti = 0;
  for (uint64_t i = 0; i < threads; i++) {
    ti += workers[i]->getElapsedTime();
  }
  return ti / threads;
}