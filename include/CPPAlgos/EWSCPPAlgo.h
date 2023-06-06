/*! \file EWSCPPAlgo.h*/
//
// Created by haolan on 5/29/23.
//

#ifndef INTELLISTREAM_EWSCPPALGO_H
#define INTELLISTREAM_EWSCPPALGO_H
#include <CPPAlgos/AbstractCPPAlgo.h>

namespace AMMBench {
/**
 * @ingroup AMMBENCH_CppAlgos The algorithms written in c++
 * @{
 */
/**
 * @class EWSCPPAlgo  CPPAlgos/EWSCPPAlgo.h
 * @brief The Element Wise Sampling (EWS) class of c++ algos
 *
 */
class EWSCPPAlgo : public AMMBench::AbstractCPPAlgo {
 public:
  EWSCPPAlgo() {

  }

  ~EWSCPPAlgo() {

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
 * @brief The class to describe a shared pointer to @ref EWSCppAlgo

 */
typedef std::shared_ptr<class AMMBench::EWSCPPAlgo> EWSCPPAlgoPtr;
/**
 * @ingroup AMMBENCH_CppAlgos
 * @def newEWSCppAlgo
 * @brief (Macro) To creat a new @ref  EWSCppAlgounder shared pointer.
 */
#define newEWSCPPAlgo std::make_shared<AMMBench::EWSCPPAlgo>
}
/**
 * @}
 */
#endif //INTELLISTREAM_EWSCPPALGO_H
