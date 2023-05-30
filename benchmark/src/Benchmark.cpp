/*! \file Benchmark.h*/

/**
 * @brief This is the main entry point of the entire program.
 * We use this as the entry point for benchmarking.
 */
#include <AMMBench.h>
#include <Utils/UtilityFunctions.h>
using namespace std;
using namespace INTELLI;
using namespace torch;
using namespace DIVERSE_METER;
void runSingleThreadTest(std::string configName) {
  MeterTable meterTable;
  AbstractMeterPtr eMeter = nullptr;
  ConfigMapPtr cfg = newConfigMap();
  cfg->fromFile(configName);
  AMMBench::MatrixLoaderTable mLoaderTable;
  uint64_t sketchDimension;
  sketchDimension = cfg->tryU64("sketchDimension", 50, true);
  uint64_t coreBind = cfg->tryU64("coreBind", 0, true);
  uint64_t usingMeter = cfg->tryU64("usingMeter", 0, true);
  std::string meterTag = cfg->tryString("meterTag", "intelMsr", true);
  uint64_t useCPP = cfg->tryU64("useCPP", 0, true);
  if (usingMeter) {
    eMeter = meterTable.findMeter(meterTag);
    if (eMeter != nullptr) {
      eMeter->setConfig(cfg);
      double staticPower = cfg->tryDouble("staticPower", 0.0, false);
      if (staticPower == 0.0) {
        eMeter->testStaticPower(2);
      } else {
        INTELLI_INFO("use pre-defined static power");
        eMeter->setStaticPower(staticPower);
      }
      INTELLI_INFO("static power is " + to_string(eMeter->getStaticPower()) + " W");
    } else {
      INTELLI_ERROR("No meter found: " + meterTag);
    }

  }
  UtilityFunctions::bind2Core((int) coreBind);
  //torch::set_num_threads(1);
  std::string ptFile = cfg->tryString("ptFile", "torchscripts/FDAMM.pt", true);

  //uint64_t customResultName = cfg->tryU64("customResultName", 0, true);
  INTELLI_INFO("Place me at core" + to_string(coreBind));
  INTELLI_INFO(
      "with sketch" + to_string(sketchDimension));
  torch::jit::script::Module module;
  INTELLI_INFO("Try pt file " + ptFile);
  module = torch::jit::load(ptFile);
  std::string matrixLoaderTag = cfg->tryString("matrixLoaderTag", "random", true);
  auto matLoaderPtr = mLoaderTable.findMatrixLoader(matrixLoaderTag);
  assert(matLoaderPtr);
  matLoaderPtr->setConfig(cfg);
  auto A = matLoaderPtr->getA();
  auto B = matLoaderPtr->getB();
  torch::Tensor C;
  //555
  /*torch::manual_seed(114514);
//555
auto A = torch::rand({(long) aRow, (long) aCol});
auto B = torch::rand({(long) aCol, (long) bCol});*/
  INTELLI_INFO("Generation done, conducting...");
  uint64_t threads = cfg->tryU64("threads", 0, true);
  ThreadPerf pef(-1);
  pef.setPerfList();
  AMMBench::BlockPartitionRunner br;
  if (threads > 1) {
    INTELLI_WARNING("use multithread");
    br.setConfig(cfg);
    br.createABC(A, B);
    if (eMeter != nullptr) {
      eMeter->startMeter();
    }
    pef.start();
    C = br.parallelForward();
    pef.end();
    if (eMeter != nullptr) {
      eMeter->stopMeter();
    }
  } else {
    AMMBench::CPPAlgoTable cppAlgoTable;
    std::string cppAlgoTag = cfg->tryString("cppAlgoTag", "mm", true);
    AMMBench::AbstractCPPAlgoPtr cppAlgoPtr = cppAlgoTable.findCppAlgo(cppAlgoTag);
    INTELLI_WARNING("single thread");
    if (eMeter != nullptr) {
      eMeter->startMeter();
    }
    pef.start();
    if (useCPP && cppAlgoPtr) {
      INTELLI_WARNING("this is pure c++");
      C = cppAlgoPtr->amm(A, B, sketchDimension);
    } else {
      C =module.forward({A, B, (long) sketchDimension}).toTensor();
    }
    pef.end();
    if (eMeter != nullptr) {
      eMeter->stopMeter();
    }

  }

  std::string ruName = "default";

  auto resultCsv = pef.resultToConfigMap();
  if (eMeter != nullptr) {
    eMeter->stopMeter();
    double energyConsumption = eMeter->getE();
    double staticEnergyConsumption = eMeter->getStaicEnergyConsumption(resultCsv->tryU64("perfElapsedTime", 0, false));
    double pureEnergy = energyConsumption - staticEnergyConsumption;
    resultCsv->edit("energyAll", (double) energyConsumption);
    resultCsv->edit("energyOnlyMe", (double) pureEnergy);
  }
  if (threads > 1) {
    INTELLI_WARNING("consider multithread elapsed time");
    resultCsv->edit("perfElapsedTime", (uint64_t) br.getElapsedTime());
  }
  // error
  INTELLI_WARNING("evaluating the error, may takes some time");
  torch::Tensor realC = torch::matmul(A, B);
  double froError = INTELLI::UtilityFunctions::relativeFrobeniusNorm(realC, C);
  double froBNormal = B.norm().item<double>();
  double errorBoundRatio = froError / froBNormal;
  INTELLI_INFO("B normal is " + to_string(froBNormal));
  resultCsv->edit("froError", (double) froError);
  resultCsv->edit("errorBoundRatio", (double) errorBoundRatio);
  resultCsv->toFile(ruName + ".csv");
  INTELLI_INFO("Done. here is result");
  std::cout << resultCsv->toString() << endl;
}

int main(int argc, char **argv) {
  string configName, outPrefix = "";
  if (argc >= 2) {
    configName += argv[1];
  } else {
    configName = "config.csv";
  }
  runSingleThreadTest(configName);
  return 0;
}

