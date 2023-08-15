#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'common')))

import csv
import numpy as np
import matplotlib.pyplot as plt
import accuBar as accuBar
import groupBar2 as groupBar2
import groupLine as groupLine
from autoParase import *
import itertools as it

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pylab
from matplotlib.font_manager import FontProperties
from matplotlib import ticker
from matplotlib.ticker import LogLocator, LinearLocator

import pandas as pd
from OoOCommon import *
from algorithms import *

OPT_FONT_NAME = 'Helvetica'
TICK_FONT_SIZE = 22
LABEL_FONT_SIZE = 22
LEGEND_FONT_SIZE = 22
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


def runPeriod(exePath, srcA,srcB, algo, resultPath, configTemplate="config.csv",prefixTag="null"):
    # resultFolder="periodTests"
    configFname = "config_period_"+prefixTag + ".csv"
    # configTemplate = "config.csv"
    # clear old files
    os.system("cd " + exePath + "&& sudo rm *.csv")
    os.system("cp perfListEvaluation.csv " + exePath)

    df = algo.config.copy()
    df.loc["srcA"]  = [srcA, 'String']
    df.loc["srcB"]  = [srcB, 'String']
    # editConfig(configTemplate, exePath + configFname, "earlierEmitMs", 0)
    editConfig(configTemplate, exePath + configFname, df)
    # prepare new file
    # run
    os.system("cd " + exePath + "&& sudo env OMP_NUM_THREADS=1 ./benchmark " + configFname)
    # copy result
    cleanPath(resultPath + "/" + str(singleValue))
    os.system("cd " + exePath + "&& sudo cp *.csv " + resultPath + "/" + str(prefixTag))


def runPeriodVector (exePath,periodVec,pS,algo,resultPath,prefixTag, configTemplate="config.csv"):
    for i in  range(len(periodVec)):
        rf=periodVec[i]
        sf=pS[i]
        print(sf)
        runPeriod(exePath, rf,sf,algo, resultPath, configTemplate,prefixTag[i])


def readResultSingle(singleValue, resultPath):
    resultFname = resultPath + "/" + str(singleValue) + "/result_streaming.csv"
    elapsedTime = readConfig(resultFname, "perfElapsedTime")
    froError = readConfig(resultFname, "froError")
    errorBoundRatio = readConfig(resultFname, "errorBoundRatio")
    thr=readConfig(resultFname, "throughputByElements")
    return elapsedTime, froError, errorBoundRatio,thr


def readResultVector(singleValueVec, resultPath):
    elapseTimeVec = []
    froErrorVec = []
    errorBoundRatioVec = []
    thrVec=[]
    for i in singleValueVec:
        elapsedTime, froError, errorBoundRatio,thr = readResultSingle(i, resultPath)
        elapseTimeVec.append(float(elapsedTime) / 1000.0)
        froErrorVec.append(float(froError))
        errorBoundRatioVec.append(float(errorBoundRatio))
        thrVec.append(float(thr))
    return np.array(elapseTimeVec), np.array(froErrorVec), np.array(
        errorBoundRatioVec),np.array(thrVec)


def compareMethod(exeSpace, commonPathBase, csvTemplate, srcAVec,srcBVec,algos,dataSetName,reRun=1):
    elapsedTimeAll = []
    periodAll = []
    froAll = []
    errorBoundRatioAll = []
    thrAll=[]
    for algo in algos:
        resultPath = commonPathBase + algo.resultPath
        if (reRun == 1):
            cleanPath(resultPath)
            runPeriodVector(exeSpace, srcAVec,srcBVec,algo, resultPath, dataSetName,csvTemplate)
        #exit()
        elapsedTime, fro, eb,thr = readResultVector(dataSetName, resultPath)
        elapsedTimeAll.append(elapsedTime)
        periodAll.append(dataSetName)
        froAll.append(fro)
        errorBoundRatioAll.append(eb)
        thrAll.append(thr)
    return np.array(elapsedTimeAll), np.array(froAll), np.array(errorBoundRatioAll),np.array(thrAll),periodAll
        
def draw2yBar(NAME,R1,R2,l1,l2,fname):
    fig, ax1 = plt.subplots(figsize=(10,4)) 
    width = 0.2  # 柱形的宽度  
    x1_list = []
    x2_list = []
    bars=[]
    index = np.arange(len(NAME))
    for i in range(len(R1)):
        x1_list.append(i)
        x2_list.append(i + width)
    #ax1.set_ylim(0, 1)
    bars.append(ax1.bar(x1_list, R1, width=width, color=COLOR_MAP[0], hatch=PATTERNS[0], align='edge',edgecolor='black', linewidth=3))
    ax1.set_ylabel("ms",fontproperties=LABEL_FP)
    ax1.set_xticklabels(ax1.get_xticklabels())  # 设置共用的x轴
    ax2 = ax1.twinx()
    
    #ax2.set_ylabel('latency/us')
    #ax2.set_ylim(0, 0.5)
    bars.append(ax2.bar(x2_list, R2, width=width,  color=COLOR_MAP[1], hatch=PATTERNS[1], align='edge', tick_label=NAME,edgecolor='black', linewidth=3))
  
    ax2.set_ylabel("%",fontproperties=LABEL_FP)
    # plt.grid(axis='y', color='gray')

    #style = dict(size=10, color='black')
    #ax2.hlines(tset, 0, x2_list[len(x2_list)-1]+width, colors = "r", linestyles = "dashed",label="tset") 
    #ax2.text(4, tset, "$T_{set}$="+str(tset)+"us", ha='right', **style)
    if (1):
        plt.legend(bars, [l1,l2],
                   prop=LEGEND_FP,
                   ncol=2,
                   loc='upper center',
                   #                     mode='expand',
                   shadow=False,
                   bbox_to_anchor=(0.55, 1.45),
                   columnspacing=0.1,
                   handletextpad=0.2,
                borderaxespad=-1,
                   #                     bbox_transform=ax.transAxes,
                   #                     frameon=True,
                   #                     columnspacing=5.5,
                   #                     handlelength=2,
                   )
    plt.xlabel(NAME, fontproperties=LABEL_FP)
    plt.xticks(size=TICK_FONT_SIZE)
    ax1.yaxis.set_major_locator(LinearLocator(5))
    ax2.yaxis.set_major_locator(LinearLocator(5))
    ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    plt.tight_layout()
    plt.savefig(fname+".pdf")

def main():
    exeSpace = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/"
    commonBasePath = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/results/AMME2E_Static_lazy/"

    figPath = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/figures/AMME2E_Static_lazy/"
    
    # add the datasets here
    srcAVec=["datasets/AST/mcfe.mtx","datasets/DWAVE/dwa512.mtx",'datasets/ECO/wm2.mtx','datasets/QCD/qcda_small.mtx','datasets/RDB/rdb2048.mtx','datasets/UTM/utm1700a.mtx','datasets/ZENIOS/zenios.mtx']
    srcBVec=["datasets/AST/mcfe.mtx","datasets/DWAVE/dwb512.mtx",'datasets/ECO/wm3.mtx','datasets/QCD/qcdb_small.mtx','datasets/RDB/rdb2048l.mtx','datasets/UTM/utm1700b.mtx','datasets/ZENIOS/zenios.mtx']
    dataSetNames=['AST','DWAVE','ECO','QCD','RDB','UTM','ZENIOS']

    # add the algo tag here
    algos = [MM_CPP, CRS_CPP]
    methodTags = list(map(lambda algo: algo.name, algos))
    # this template configs all algos as lazy mode, all datasets are static and normalized    
    csvTemplate = 'config_e2e_sl.csv'

    # run
    reRun = 0
    if (len(sys.argv) < 2):
        os.system("mkdir ../../results")
        os.system("mkdir ../../figures")
        os.system("mkdir " + figPath)
        os.system("sudo rm -rf " + commonBasePath)
        os.system("sudo mkdir " + commonBasePath)
        reRun = 1

    lat95All, errAll, ebAll,thrAll,periodAll = compareMethod(exeSpace, commonBasePath, csvTemplate, srcAVec,srcBVec,algos,dataSetNames, reRun)
    print(lat95All[0][0])
    errAll=np.array(errAll)*100.0
    #draw2yBar(methodTags,[lat95All[0][0],lat95All[1][0],lat95All[2][0],lat95All[3][0]],[errAll[0][0],errAll[1][0],errAll[2][0],errAll[3][0]],'95% latency (ms)','Error (%)',figPath + "sec6_5_stock_q1_normal")
    groupBar2.DrawFigure(dataSetNames, np.array(errAll), methodTags, "Datasets", "Error (%)",
                         5, 15, figPath + "sec4_1_e2e_sl_fro", True)
    groupBar2.DrawFigure(dataSetNames, np.array(lat95All), methodTags, "Datasets", "95% latency (ms)",
                         5, 15, figPath + "sec4_1_e2e_sl_latency", False)
    groupBar2.DrawFigure(dataSetNames, np.array(thrAll)/1000.0, methodTags, "Datasets", "elements/ms",
                         5, 15, figPath + "sec4_1_e2e_sl_throughput", False)
    print(lat95All,errAll)
if __name__ == "__main__":
    main()
