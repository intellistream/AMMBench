#include <vector>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include <AMMBench.h>
using namespace std;
using namespace INTELLI;
using namespace torch;

TEST_CASE("Test Block LRA in cpp", "[short]")
{
    torch::manual_seed(114514);
    AMMBench::BlockLRACPPAlgo wcr;
    auto A = torch::rand({500, 400});
    auto B = torch::rand({400, 600});
    auto realC = torch::matmul(A, B);
    auto ammC = wcr.amm(A, B, 100, 0.5, 0.5);
    double froError = INTELLI::UtilityFunctions::relativeFrobeniusNorm(realC, ammC);
    std::cout << "froError: " << froError << std::endl;
    REQUIRE(froError < 0.5);
}