//
// Created by tony on 25/05/23.
//

#include <CPPAlgos/CPPAlgoTable.h>
#include <CPPAlgos/CRSCPPAlgo.h>
#include <CPPAlgos/CRSV2CPPAlgo.h>
namespace AMMBench {
AMMBench::CPPAlgoTable::CPPAlgoTable() {
  algoMap["mm"] = newAbstractCPPAlgo();
  algoMap["crs"] = newCRSCPPAlgo();
    algoMap["crsV2"] = newCRSV2CPPAlgo();
}

} // AMMBench