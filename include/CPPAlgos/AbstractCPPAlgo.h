/*! \file AbstractCPPAlgo.h*/
//
// Created by tony on 25/05/23.
//

#ifndef INTELLISTREAM_INCLUDE_CPPALGOS_ABSTRACTCPPALGO_H_
#define INTELLISTREAM_INCLUDE_CPPALGOS_ABSTRACTCPPALGO_H_
#include <Utils/AbstractC20Thread.hpp>
#include <Utils/ConfigMap.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <memory>
#include <vector>
namespace AMMBench {
/**
 * @ingroup AMMBENCH_CppAlgos The algorithms writtrn in c++
 * @{
 */
/**
 * @class AbstractCPPAlgo CPPAlgos/AbstractCPPAlgo.h
 * @brief The abstract class of c++ algos
 */
class AbstractCPPAlgo {
 public:
  AbstractCPPAlgo() {

  }
  ~AbstractCPPAlgo() {

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
 * @brief The class to describe a shared pointer to @ref AbstractCPPAlgo

 */
typedef std::shared_ptr<class AMMBench::AbstractCPPAlgo> AbstractCPPAlgoPtr;
/**
 * @ingroup AMMBENCH_CppAlgos
 * @def newAbstractCppAlgo
 * @brief (Macro) To creat a new @ref  AbstractCppAlgounder shared pointer.
 */
#define newAbstractCPPAlgo std::make_shared<AMMBench::AbstractCPPAlgo>
}
/**
 * @}
 */

#endif //INTELLISTREAM_INCLUDE_CPPALGOS_ABSTRACTCPPALGO_H_
