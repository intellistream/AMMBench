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

scanTag = "aRow"


def singleRun(exePath, singleValue, resultPath, configTemplate):
    # resultFolder="singleValueTests"
    configFname = "config_" + scanTag + str(singleValue) + ".csv"
    # configTemplate = "config.csv"
    # clear old files
    os.system("cd " + exePath + "&& sudo rm *.csv")

    # editConfig(configTemplate, exePath + configFname, "earlierEmitMs", 0)
    editConfig(configTemplate, exePath + configFname, scanTag, singleValue)
    # prepare new file
    # run
    os.system("cd " + exePath + "&& sudo env OMP_NUM_THREADS=1 ./benchmark " + configFname)
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
    cacheMiss = readConfig(resultFname, "cacheMiss")
    cacheRefs = readConfig(resultFname, "cacheRefs")
    return elapsedTime, cacheMiss, cacheRefs


def cleanPath(path):
    os.system("sudo rm -rf " + path)
    os.system("sudo mkdir " + path)


def readResultVector(singleValueVec, resultPath):
    elapseTimeVec = []
    cacheMissVec = []
    cacheRefVec = []
    for i in singleValueVec:
        elapsedTime, cacheMiss, cacheRefs = readResultSingle(i, resultPath)
        elapseTimeVec.append(float(elapsedTime) / 1000.0)
        cacheMissVec.append(float(cacheMiss))
        cacheRefVec.append(float(cacheRefs))
    return np.array(elapseTimeVec), np.array(cacheMissVec), np.array(cacheRefVec)


def compareMethod(exeSpace, commonPathBase, resultPaths, csvTemplates, periodVec, reRun=1):
    elapsedTimeAll = []
    cacheMissAll = []
    cacheRefAll = []
    periodAll = []
    for i in range(len(csvTemplates)):
        resultPath = commonPathBase + resultPaths[i]
        if (reRun == 1):
            os.system("sudo rm -rf " + resultPath)
            os.system("sudo mkdir " + resultPath)
            runScanVector(exeSpace, periodVec, resultPath, csvTemplates[i])
        elapsedTime, cacheMiss, cacheRef = readResultVector(periodVec, resultPath)
        elapsedTimeAll.append(elapsedTime)
        cacheMissAll.append(cacheMiss)
        cacheRefAll.append(cacheRef)
        periodAll.append(periodVec)
        cacheMissRateAll = np.array(cacheMissAll) / np.array(cacheRefAll) * 100.0
        # periodAll.append(periodVec)
    return np.array(elapsedTimeAll), cacheMissRateAll, periodAll


def main():
    exeSpace = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/"
    commonBase = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/results/" + scanTag + "/"
    figPath = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/figures/" + scanTag
    methodTags = ["Co-AMM", "BCo-AMM", "Count-sketch", "Tug-of-War", "MM"]
    resultPaths = ["co", "co", "cs", "tow", "mm"]
    csvTemplates = ["config_CoAMM.csv", "config_BCoAMM.csv", "config_CounterSketch.csv",
                    "config_TugOfWar.csv", "config_RAWMM.csv"]
    valueVec = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
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
    elapsedTimeAll, cacheMissAll, periodAll = compareMethod(exeSpace, commonBase, resultPaths, csvTemplates, valueVec,
                                                            reRun)
    groupLine.DrawFigure(periodAll, elapsedTimeAll,
                         methodTags,
                         "#elements in A's row", "elapsed time (ms)", 0, 1,
                         figPath + "/" + scanTag + "sketch_elapsedTime",
                         True)
    groupLine.DrawFigureYnormal(periodAll, cacheMissAll,
                                methodTags,
                                "#elements in A's row", "cacheMiss (%)", 0, 1,
                                figPath + "/" + scanTag + "sketch_cacheMiss",
                                True)
    # sampling
    resultPaths = ["crs", "crsv2", "bcrs", "wcr", "ews", "mm"]
    csvTemplates = ["config_CRS.csv", "config_CRSV2.csv", "config_BerCRS.csv", "config_WCR.csv",
                    "config_EWS.csv", "config_RAWMM.csv"]
    methodTags = ["CRS", "CRSV2", "Ber-CRS", "Weighted-CR", "EWS", "MM"]
    elapsedTimeAll, cacheMissAll, periodAll = compareMethod(exeSpace, commonBase, resultPaths, csvTemplates, valueVec,
                                                            reRun)
    groupLine.DrawFigure(periodAll, elapsedTimeAll,
                         methodTags,
                         "#elements in A's row", "elapsed time (ms)", 0, 1,
                         figPath + "/" + scanTag + "sampling_elapsedTime",
                         True)
    groupLine.DrawFigureYnormal(periodAll, cacheMissAll,
                                methodTags,
                                "#elements in A's row", "cacheMiss (%)", 0, 1,
                                figPath + "/" + scanTag + "sampling_cacheMiss",
                                True)
    # draw2yLine("watermark time (ms)",singleValueVecDisp,lat95Vec,errVec,"95% Latency (ms)","Error","ms","",figPath+"wm_lat")
    # draw2yLine("watermark time (ms)",singleValueVecDisp,thrVec,errVec,"Throughput (KTp/s)","Error","KTp/s","",figPath+"wm_thr")
    # draw2yLine("watermark time (ms)",singleValueVecDisp,lat95Vec,compVec,"95% Latency (ms)","Completeness","ms","",figPath+"wm_omp")
    # groupLine.DrawFigureYnormal([singleValueVec,singleValueVec],[errVec,aqpErrVec],['w/o aqp',"w/ MeanAqp"],"watermark time (ms)","Error",0,1,figPath+"wm_MeanAqp",True)
    # print(errVec)
    # print(aqpErrVec)
    # print(elapseTimeVecFD)
    # readResultsingleValue(50,resultPath)


if __name__ == "__main__":
    main()
