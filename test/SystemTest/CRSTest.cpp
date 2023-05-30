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
  uint64_t sketchDimension;
  sketchDimension = cfg->tryU64("sketchDimension", 50, true);
  uint64_t coreBind = cfg->tryU64("coreBind", 0, true);
  UtilityFunctions::bind2Core((int) coreBind);
  torch::set_num_threads(1);
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
  /*torch::manual_seed(114514);
//555
auto A = torch::rand({(long) aRow, (long) aCol});
auto B = torch::rand({(long) aCol, (long) bCol});*/
  INTELLI_INFO("Generation done, conducting...");
  ThreadPerf pef((int) coreBind);
  pef.setPerfList();
  pef.start();
  auto C =module.forward({A, B, (long) sketchDimension}).toTensor();
  pef.end();
  std::string ruName = "default";

  auto resultCsv = pef.resultToConfigMap();
  resultCsv->toFile(ruName + ".csv");
  INTELLI_INFO("Done. here is result");
  std::cout << resultCsv->toString() << endl;
}
TEST_CASE("Test the COLUMN ROW SAMPLINGS", "[short]")
{
  int a = 0;
  runSingleThreadTest("scripts/config_CRS.csv");
  // place your test here
  REQUIRE(a == 0);
}
TEST_CASE("Test the COLUMN ROW SAMPLINGS, V2", "[short]")
{
  int a = 0;
  runSingleThreadTest("scripts/config_CRSV2.csv");
  // place your test here
  REQUIRE(a == 0);
}
TEST_CASE("Test CRS in cpp", "[short]")
{
  torch::manual_seed(114514);
  AMMBench::CRSCPPAlgo crs;
  auto A = torch::rand({400, 400});
  auto B = torch::rand({400, 400});
  auto realC = torch::matmul(A, B);
  auto ammC = crs.amm(A, B, 20);
  double froError = INTELLI::UtilityFunctions::relativeFrobeniusNorm(realC, ammC);
  REQUIRE(froError < 0.5);
}
TEST_CASE("Test CRS v2 in cpp", "[short]")
{
    torch::manual_seed(114514);
    AMMBench::CRSV2CPPAlgo crs;
    auto A = torch::rand({400, 400});
    auto B = torch::rand({400, 400});
    auto realC = torch::matmul(A, B);
    auto ammC = crs.amm(A, B, 20);
    double froError = INTELLI::UtilityFunctions::relativeFrobeniusNorm(realC, ammC);
    REQUIRE(froError < 0.5);
}