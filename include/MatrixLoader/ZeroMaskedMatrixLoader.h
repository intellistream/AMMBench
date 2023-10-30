/*! \file ZeroMaskedMatrixLoader.h*/
//
// Created by tony on 10/05/23.
//

#ifndef INTELLISTREAM_INCLUDE_MATRIXLOADER_ZEROMASKEDMATRIXLOADER_H_
#define INTELLISTREAM_INCLUDE_MATRIXLOADER_ZEROMASKEDMATRIXLOADER_H_

#include <MatrixLoader/AbstractMatrixLoader.h>

namespace AMMBench {
/**
 * @ingroup AMMBENCH_MatrixLOADER
 * @{
 */
/**
 * @ingroup AMMBENCH_MatrixLOADER_zero masked The  zero-masked Random generator
 * @{
 */
/**
 * @class ZeroMaskedMatrixLoader MatrixLoader/ZeroMaskedMatrixLoader.h
 * @brief The zero masked class of matrix loader, given generate a n*m matrix, where only the left-top n1*m2 contents are not zero
 * @ingroup AMMBENCH_MatrixLOADER_Random
 * @note:
 * - Must have a global config by @ref setConfig
 * @note  Default behavior
* - create
* - call @ref setConfig, this function will also generate the tensor A and B correspondingly
* - call @ref getA and @ref getB (assuming we are benchmarking torch.mm(A,B))
 * @note: require config parameters and default values
 * - "aRow" The rows in matrix A, U64, 100
 * - "aCol" The cols in matrix B, U64, 1000
 * - "bCol" The rows in matrix B, U64, 500
 * - "seed" The seed of inline random generator,U64,114514
 * - "nnzA" the ratio of nnz values in matrix A, Double, 1.0
 * - "nnzB" the ratio of nnz values in matrix B, Double, 1.0
 * @note: default name tags
 * "zeroMasked": @ref ZeroMaskedMatrixLoader
 */
class ZeroMaskedMatrixLoader : public AbstractMatrixLoader {
 protected:
  torch::Tensor A, B;
  uint64_t aRow, aCol, bCol, seed;
  double nnzA,nnzB;

  /**
   * @brief Inline logic of reading a config file
   * @param cfg the config
   */
  void paraseConfig(INTELLI::ConfigMapPtr cfg);

  /**
   * @brief inline logic of generating A and B
   */
  void generateAB();

 public:
  ZeroMaskedMatrixLoader() = default;

  ~ZeroMaskedMatrixLoader() = default;

  /**
     * @brief Set the GLOBAL config map related to this loader
     * @param cfg The config map
      * @return bool whether the config is successfully set
      * @note
     */
  virtual bool setConfig(INTELLI::ConfigMapPtr cfg);

  /**
   * @brief get the A matrix
   * @return the generated A matrix
   */
  virtual torch::Tensor getA();

  /**
  * @brief get the B matrix
  * @return the generated B matrix
  */
  virtual torch::Tensor getB();
};

/**
 * @ingroup AMMBENCH_MatrixLOADER_Random
 * @typedef ZeroMaskedMatrixLoaderPtr
 * @brief The class to describe a shared pointer to @ref ZeroMaskedMatrixLoader

 */
typedef std::shared_ptr<class AMMBench::ZeroMaskedMatrixLoader> ZeroMaskedMatrixLoaderPtr;
/**
 * @ingroup AMMBENCH_MatrixLOADER_Random
 * @def newZeroMaskedMatrixLoader
 * @brief (Macro) To creat a new @ref ZeroMaskedMatrixLoader under shared pointer.
 */
#define newZeroMaskedMatrixLoader std::make_shared<AMMBench::ZeroMaskedMatrixLoader>
/**
 * @}
 */
/**
 * @}
 */
}
#endif //INTELLISTREAM_INCLUDE_MATRIXLOADER_RANDOMMATRIXLOADER_H_
