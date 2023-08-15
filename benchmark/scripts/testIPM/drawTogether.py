#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'common')))

import csv
import numpy as np
import accuBar as accuBar
import groupBar as groupBar
import groupBar2 as groupBar2
import groupLine as groupLine
from autoParase import *
import itertools as it

import matplotlib
import numpy as np
import pylab
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import LogLocator, LinearLocator
import pandas as pd
from OoOCommon import *
import time
from algorithms import *

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


def singleRun(exePath, singleValue, resultPath, configTemplate, algo):
    # resultFolder="singleValueTests"
    configFname = "config_" + scanTag + str(singleValue) + ".csv"
    # configTemplate = "config.csv"
    # clear old files
    os.system("cd " + exePath + "&& sudo rm *.csv")

    df = algo.config.copy()
    df.loc[scanTag] = [singleValue, 'U64']
    # editConfig(configTemplate, exePath + configFname, "earlierEmitMs", 0)
    editConfig(configTemplate, exePath + configFname, df)
    # prepare new file
    # run
    os.system("cd " + exePath + "&& sudo env OMP_NUM_THREADS=1 ./benchmark " + configFname)
    # copy result
    cleanPath(resultPath + "/" + str(singleValue))
    cleanPath(resultPath + "/" + str(singleValue))
    os.system("cd " + exePath + "&& sudo cp *.csv " + resultPath + "/" + str(singleValue))


def runScanVector(exePath, singleValueVec, resultPath, templateName, algo):
    for i in singleValueVec:
        singleRun(exePath, i, resultPath, templateName, algo)


def readResultSingle(singleValue, resultPath):
    resultFname = resultPath + "/" + str(singleValue) + "/default.csv"
    elapsedTime = readConfig(resultFname, "perfElapsedTime")
    cacheMiss = readConfig(resultFname, "cacheMiss")
    cacheRefs = readConfig(resultFname, "cacheRefs")
    froError = readConfig(resultFname, "froError")
    instructions = readConfig(resultFname, "instructions")
    errorBoundRatio = readConfig(resultFname, "errorBoundRatio")
    return elapsedTime, cacheMiss, cacheRefs, froError, errorBoundRatio, instructions


def readResultVector(singleValueVec, resultPath):
    elapseTimeVec = []
    cacheMissVec = []
    cacheRefVec = []
    froErrorVec = []
    errorBoundRatioVec = []
    instructionVec = []
    for i in singleValueVec:
        elapsedTime, cacheMiss, cacheRefs, froError, errorBoundRatio, ins = readResultSingle(i, resultPath)
        elapseTimeVec.append(float(elapsedTime) / 1000.0)
        cacheMissVec.append(float(cacheMiss))
        cacheRefVec.append(float(cacheRefs))
        froErrorVec.append(float(froError))
        errorBoundRatioVec.append(float(errorBoundRatio))
        instructionVec.append(float(ins))
    return np.array(elapseTimeVec), np.array(cacheMissVec), np.array(cacheRefVec), np.array(froErrorVec), np.array(
        errorBoundRatioVec), np.array(instructionVec)


def compareMethod(exeSpace, commonPathBase, algos, csvTemplate, periodVec, reRun=1):
    elapsedTimeAll = []
    cacheMissAll = []
    cacheRefAll = []
    periodAll = []
    froAll = []
    errorBoundRatioAll = []
    insAll = []
    memAll = []
    for algo in algos:
        resultPath = commonPathBase + algo.resultPath
        if (reRun == 1):
            cleanPath(resultPath)
            runScanVector(exeSpace, periodVec, resultPath, csvTemplate, algo)
        elapsedTime, cacheMiss, cacheRef, fro, eb, insv = readResultVector(periodVec, resultPath)
        elapsedTimeAll.append(elapsedTime)
        cacheMissAll.append(cacheMiss)
        cacheRefAll.append(cacheRef)
        periodAll.append(periodVec)
        cacheMissRateAll = np.array(cacheMissAll) / np.array(cacheRefAll) * 100.0
        froAll.append(fro)
        errorBoundRatioAll.append(eb)
        insAll.append(insv)
        memAll.append(cacheRef)
        # periodAll.append(periodVec)
    return np.array(elapsedTimeAll), cacheMissRateAll, periodAll, np.array(froAll), np.array(
        errorBoundRatioAll), np.array(insAll), np.array(memAll)


def main():
    exeSpace = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/"
    commonBase = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/results/" + scanTag + "IPM/"
    figPath = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/figures/" + scanTag + "CPPIPM"

    algos = [CRS_CPP, SMP_PCA_CPP, COOFD_CPP, MM_CPP]
    methodTags = list(map(lambda algo: algo.name, algos))

    csvTemplate = "config.csv"
    valueVec = [100, 200, 500, 1000, 2000, 5000]
    # run
    reRun = 0
    if (len(sys.argv) < 2):
        os.system("mkdir ../../results")
        os.system("mkdir ../../figures")
        os.system("sudo rm -rf " + commonBase)
        os.system("sudo mkdir " + commonBase)
        os.system("mkdir " + figPath)
        reRun = 1

    # sampling
    elapsedTimeAll, cacheMissAll, periodAll, fro, eb, ins, mem = compareMethod(exeSpace, commonBase,
                                                                     algos, csvTemplate, valueVec, reRun)

    groupLine.DrawFigureYnormal(periodAll,
                                ins,
                                methodTags,
                                "#A's row", "instructions", 0, 100, figPath + "/" + scanTag
                                + "sampling_cpp_ins",
                                True)
    groupLine.DrawFigureYnormal(periodAll,
                                mem,
                                methodTags,
                                "#A's row", "mem access", 0, 100, figPath + "/" + scanTag
                                + "sampling_cpp_mem",
                                True)
    groupLine.DrawFigureYnormal(periodAll,
                                ins / mem,
                                methodTags,
                                "#A's row", "ipm", 0, 100, figPath + "/" + scanTag
                                + "sampling_cpp_ipm",
                                True)

    algos = [CRS_CPP, SMP_PCA_CPP, MM_CPP]
    methodTags = list(map(lambda algo: algo.name, algos))
    elapsedTimeAll, cacheMissAll, periodAll, fro, eb, ins, mem = compareMethod(exeSpace, commonBase,
                                                                     algos, csvTemplate, valueVec, reRun)

    groupLine.DrawFigureYnormal(periodAll,
                                ins,
                                methodTags,
                                "#A's row", "instructions", 0, 100, figPath + "/" + scanTag
                                + "nocofd_cpp_ins",
                                True)
    groupLine.DrawFigureYnormal(periodAll,
                                mem,
                                methodTags,
                                "#A's row", "mem access", 0, 100, figPath + "/" + scanTag
                                + "nocdfd_cpp_mem",
                                True)
    groupLine.DrawFigureYnormal(periodAll,
                                ins / mem,
                                methodTags,
                                "#A's row", "ipm", 0, 100, figPath + "/" + scanTag
                                + "nocofd_cpp_ipm",
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
