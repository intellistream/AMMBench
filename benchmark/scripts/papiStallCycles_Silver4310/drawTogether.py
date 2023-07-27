#!/usr/bin/env python3
import csv
import numpy as np
import accuBar as accuBar
import groupBar as groupBar
import groupBar2 as groupBar2
import groupLine as groupLine
from autoParase import *
import itertools as it
import os

import matplotlib
import numpy as np
import pylab
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import LogLocator, LinearLocator
import os
import pandas as pd
import sys
from OoOCommon import *
import time

# OPT_FONT_NAME = 'Helvetica'
TICK_FONT_SIZE = 22
LABEL_FONT_SIZE = 28
LEGEND_FONT_SIZE = 30
LABEL_FP = FontProperties(style='normal', size=LABEL_FONT_SIZE)
LEGEND_FP = FontProperties(style='normal', size=LEGEND_FONT_SIZE)
TICK_FP = FontProperties(style='normal', size=TICK_FONT_SIZE)

MARKERS = (['*', '|', 'v', "^", "", "h", "<", ">", "+", "d", "<", "|", "", "+", "_"])
# you may want to change the color map for different figures
COLOR_MAP = (
    '#B03A2E', '#2874A6', '#239B56', '#7D3C98', '#FFFFFF', '#F1C40F', '#F5CBA7', '#82E0AA', '#AEB6BF', '#AA4499')
# you may want to change the patterns for different figures
PATTERNS = (["////", "o", "", "||", "-", "//", "\\", "o", "O", "////", ".", "|||", "o", "---", "+", "\\\\", "*"])
LABEL_WEIGHT = 'bold'
LINE_COLORS = COLOR_MAP
LINE_WIDTH = 3.0
MARKER_SIZE = 15.0
MARKER_FREQUENCY = 1000

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['xtick.labelsize'] = TICK_FONT_SIZE
matplotlib.rcParams['ytick.labelsize'] = TICK_FONT_SIZE
matplotlib.rcParams['font.family'] = OPT_FONT_NAME
matplotlib.rcParams['pdf.fonttype'] = 42

scanTag = "aCol"


def singleRun(exePath, singleValue, resultPath, configTemplate):
    # resultFolder="singleValueTests"
    configFname = "config_" + scanTag + str(singleValue) + ".csv"
    # configTemplate = "config.csv"
    # clear old files

    os.system("cd " + exePath + "&& sudo rm default*.csv config*.csv *result*.csv perf*.csv")
    os.system("cp perfListEvaluation.csv "+exePath)
    # editConfig(configTemplate, exePath + configFname, "earlierEmitMs", 0)
    editConfig(configTemplate, exePath + configFname, scanTag, singleValue)
    # prepare new file
    # run
    os.system("export OMP_NUM_THREADS=1 &&" + "cd " + exePath + "&& sudo ./benchmark " + configFname)
    # copy result
    os.system("sudo rm -rf " + resultPath + "/" + str(singleValue))
    os.system("sudo mkdir " + resultPath + "/" + str(singleValue))
    os.system("cd " + exePath + "&& sudo cp *.csv " + resultPath + "/" + str(singleValue))


def runScanVector(exePath, singleValueVec, resultPath, templateName="config.csv"):
    for i in singleValueVec:
        singleRun(exePath, i, resultPath, templateName)


def readResultSingle(singleValue, resultPath):
    resultFname = resultPath + "/" + str(singleValue) + "/default.csv"
    elapsedTime = readConfig(resultFname, "perfElapsedTime")
    memStall = readConfig(resultFname, "memStall")
    l1dStall = readConfig(resultFname, "l1dStall")
    cpuCycle = readConfig(resultFname, "cpuCycle")
    l2Stall = readConfig(resultFname, "l2Stall")
    return elapsedTime, memStall, l1dStall, cpuCycle, l2Stall


def cleanPath(path):
    os.system("sudo rm -rf " + path)
    os.system("sudo mkdir " + path)


def readResultVector(singleValueVec, resultPath):
    elapseTimeVec = []
    memStallVec = []
    l1dStallVec = []
    cpuCycleVec = []
    l2StallVec = []
    for i in singleValueVec:
        elapsedTime, memStall, l1dStall, cpuCycle, l2Stall = readResultSingle(i, resultPath)
        elapseTimeVec.append(float(elapsedTime) / 1000.0)
        memStallVec.append(float(memStall))
        l1dStallVec.append(float(l1dStall))
        cpuCycleVec.append(float(cpuCycle))
        l2StallVec.append(float(l2Stall))
    return np.array(elapseTimeVec), np.array(memStallVec), np.array(l1dStallVec), np.array(cpuCycleVec), np.array(
        l2StallVec)


def compareMethod(exeSpace, commonPathBase, resultPaths, csvTemplates, periodVec, reRun=1):
    elapsedTimeAll = []
    memStallAll = []
    l1dStallAll = []
    periodAll = []
    cpuCycleAll = []
    l2StallAll = []
    for i in range(len(csvTemplates)):
        resultPath = commonPathBase + resultPaths[i]
        if (reRun == 1):
            os.system("sudo rm -rf " + resultPath)
            os.system("sudo mkdir " + resultPath)
            runScanVector(exeSpace, periodVec, resultPath, csvTemplates[i])
        elapsedTime, memStall, l1dStall, cpuCycle, l2Stall = readResultVector(periodVec, resultPath)
        elapsedTimeAll.append(elapsedTime)
        memStallAll.append(memStall)
        l1dStallAll.append(l1dStall)
        periodAll.append(periodVec)
        memStallRateAll = np.array(memStallAll) 
        cpuCycleAll.append(cpuCycle)
        l2StallAll.append(l2Stall)
        # periodAll.append(periodVec)
    return np.array(elapsedTimeAll), np.array(memStallRateAll), np.array(periodAll), np.array(cpuCycleAll), np.array(l1dStallAll),np.array(l2StallAll)

def getCyclesPerMethod(cyclesAll,valueChose):
    cpuCyclePerMethod=[]
    for i in range(len(cyclesAll)):
        cpuCyclePerMethod.append(cyclesAll[int(i)][int(valueChose)])
    return np.array(cpuCyclePerMethod)
def main():
    exeSpace = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/"
    commonBase = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/results/" + scanTag + "_stalls/"
    figPath = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/figures/" + scanTag + "_stalls"
    methodTags = ["CRS", "MM"]
    resultPaths = ["crs", "mm"]
    csvTemplates = ["config_CPPCRS.csv", "config_CPPMM.csv"]
    valueVec = [200,500,1000,2000,5000]
    valueVecDisp = np.array(valueVec)
    # run
    reRun = 0
    if (len(sys.argv) < 2):
        os.system("mkdir ../../results")
        os.system("mkdir ../../figures")
        os.system("mkdir " + figPath)
        os.system("sudo rm -rf " + commonBase)
        os.system("sudo mkdir " + commonBase)
        reRun = 1
    # skech
    elapsedTimeAll, memStallAll, periodAll, cpuCycle, l1dStallAll,l2StallAll = compareMethod(exeSpace, commonBase, resultPaths, csvTemplates,
                                                                     valueVec,
                                                                     reRun)
    groupLine.DrawFigure(periodAll, elapsedTimeAll,
                         methodTags,
                         "#elements in A's row", "elapsed time (ms)", 0, 1,
                         figPath + "/" + scanTag + "sketch_cpp_elapsedTime",
                         True)
    groupLine.DrawFigureYnormal(periodAll, memStallAll,
                                methodTags,
                                "#elements in A's row", "mem stall cycles ", 0, 1,
                                figPath + "/" + scanTag + "_memStall",
                                True)
    groupLine.DrawFigureYnormal(periodAll,
                                cpuCycle ,
                                methodTags,
                                "#A's row", "total cycles", 0, 100, figPath + "/" + scanTag +
                                "_cpuCycle",
                                True)
    groupLine.DrawFigureYnormal(periodAll,
                                l2StallAll,
                                methodTags,
                                "#A's row", "l2 stall cycles", 0, 100, figPath + "/" + scanTag
                                + "_l2_stall",
                                True)
    otherCycle=cpuCycle-memStallAll-l1dStallAll-l2StallAll
    # draw2yLine("watermark time (ms)",singleValueVecDisp,lat95Vec,errVec,"95% Latency (ms)","Error","ms","",figPath+"wm_lat")
    # draw2yLine("watermark time (ms)",singleValueVecDisp,thrVec,errVec,"Throughput (KTp/s)","Error","KTp/s","",figPath+"wm_thr")
    # draw2yLine("watermark time (ms)",singleValueVecDisp,lat95Vec,compVec,"95% Latency (ms)","Completeness","ms","",figPath+"wm_omp")
    # groupLine.DrawFigureYnormal([singleValueVec,singleValueVec],[errVec,aqpErrVec],['w/o aqp',"w/ MeanAqp"],"watermark time (ms)","Error",0,1,figPath+"wm_MeanAqp",True)
    print(otherCycle[0],len(otherCycle))
    allowLegend=1
    for valueChose in range(len(valueVec)):
        #cpuCyclePerMethod=getCyclesPerMethod(cpuCycle,valueChose)
        memStallPerMethod=getCyclesPerMethod(memStallAll,valueChose)
        l1dStallPerMethod=getCyclesPerMethod(l1dStallAll,valueChose)
        l2StallPerMethod=getCyclesPerMethod(l2StallAll,valueChose)
        otherPerMethod=getCyclesPerMethod(otherCycle,valueChose)
        accuBar.DrawFigure(methodTags,[memStallPerMethod,l1dStallPerMethod,l2StallPerMethod,otherPerMethod],['mem stall','l1d stall','l2 stall','others'],'','cycles', figPath + "/" + scanTag
                                    + "_stall_accubar"+str(valueVecDisp[valueChose]),allowLegend,scanTag+"="+str(valueVecDisp[valueChose]))
        allowLegend=0
    # print(aqpErrVec)
    # print(elapseTimeVecFD)
    # readResultsingleValue(50,resultPath)


if __name__ == "__main__":
    main()
