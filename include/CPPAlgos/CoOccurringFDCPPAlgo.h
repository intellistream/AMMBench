/*! \file CoOFDCPPAlgo.h*/
// Created by haolan on 5/29/23.
//

#ifndef INTELLISTREAM_COOCCURRINGFDCPPALGO_H
#define INTELLISTREAM_COOCCURRINGFDCPPALGO_H

#include <CPPAlgos/AbstractCPPAlgo.h>

namespace AMMBench {
/**
 * @ingroup AMMBENCH_CppAlgos The algorithms written in c++
 * @{
 */
/**
 * @class CoOccurringFDCPPAlgo CPPAlgos/CoOccurringFDCPPAlgo.h
 * @brief The Co-Occurring FD AMM class of c++ algos
 *
 */
class CoOccurringFDCPPAlgo : public AMMBench::AbstractCPPAlgo {
 public:
  CoOccurringFDCPPAlgo() {

  }

  ~CoOccurringFDCPPAlgo() {

  }

  /**
   * @brief the virtual function provided for outside callers, rewrite in children classes
   * @param A the A matrix
   * @param B the B matrix
   * @param sketchSize the size of sketc or sampling
   * @return the output c matrix
   */
  virtual torch::Tensor amm(torch::Tensor A, torch::Tensor B, uint64_t sketchSize);

};

/**
 * @ingroup AMMBENCH_CppAlgos
 * @typedef AbstractMatrixCppAlgoPtr
 * @brief The class to describe a shared pointer to @ref CoOccurringFDCppAlgo

 */
typedef std::shared_ptr<class AMMBench::CoOccurringFDCPPAlgo> CoOccurringFDCPPAlgoPtr;
/**
 * @ingroup AMMBENCH_CppAlgos
 * @def newCoOccurringFDCppAlgo
 * @brief (Macro) To creat a new @ref  CoOccurringFDCppAlgounder shared pointer.
 */
#define newCoOccurringFDCPPAlgo std::make_shared<AMMBench::CoOccurringFDCPPAlgo>
}
/**
 * @}
 */
#endif //INTELLISTREAM_COOCCURRINGFDCPPALGO_H
