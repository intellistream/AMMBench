#include <vector>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include <AMMBench.h>
using namespace std;
using namespace INTELLI;
using namespace torch;
void runSingleThreadTest(std::string configName) {
  ConfigMapPtr cfg = newConfigMap();
  cfg->fromFile(configName);
  AMMBench::MatrixLoaderTable mLoaderTable;

  // Get value with defaults.
  uint64_t sketchDimension = cfg->tryU64("sketchDimension", 50, true);
  uint64_t coreBind = cfg->tryU64("coreBind", 0, true);

  // Bind the thead to core. -> to prevent scheduling overhead? right?
  UtilityFunctions::bind2Core((int) coreBind);
  torch::set_num_threads(1);

  // pt file is normally the model of ML. Seems to be the running result here?
  // Load path of pt.
  std::string ptFile = cfg->tryString("ptFile", "torchscripts/Quantilization/quant.pt", true);

  //uint64_t customResultName = cfg->tryU64("customResultName", 0, true);
  INTELLI_INFO("Place me at core" + to_string(coreBind));
  INTELLI_INFO(
      "with sketch" + to_string(sketchDimension));
  
  // Why jit here? Guess we are using jit module to (de)serialize the pt.
  torch::jit::script::Module module;
  INTELLI_INFO("Try pt file " + ptFile);
  module = torch::jit::load(ptFile);

  // Loader TAG. We are loading the matrix loader according to the name.
  std::string matrixLoaderTag = cfg->tryString("matrixLoaderTag", "random", true);
  auto matLoaderPtr = mLoaderTable.findMatrixLoader(matrixLoaderTag);
  assert(matLoaderPtr);
  matLoaderPtr->setConfig(cfg);
  auto A = matLoaderPtr->getA();
  auto B = matLoaderPtr->getB();
  /*torch::manual_seed(114514);
//555
auto A = torch::rand({(long) aRow, (long) aCol});
auto B = torch::rand({(long) aCol, (long) bCol});*/
  INTELLI_INFO("Generation done, conducting...");

  // TODO: How to train within cpp here?

  // Point the core to bind and pef.
  ThreadPerf pef((int) coreBind);
  pef.setPerfList();
  pef.start();
  auto C = module.forward({A, B, (long) sketchDimension}).toTensor();
  pef.end();
  std::string ruName = "default";

  // Serialize and save.
  auto resultCsv = pef.resultToConfigMap();
  resultCsv->toFile(ruName + ".csv");
  INTELLI_INFO("Done. here is result");
  std::cout << resultCsv->toString() << endl;
}
TEST_CASE("Test the COLUMN ROW SAMPLINGS", "[short]")
{
  int a = 0;
  runSingleThreadTest("scripts/config_Quantilization.csv");
  // place your test here
  REQUIRE(a == 0);
}