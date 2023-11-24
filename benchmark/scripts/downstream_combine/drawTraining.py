import sys
import os
import io
import numpy as np
import re

from run import main as runOnce
from draw import main as runAll



def main():
    time, accuracy, time2, accuracy2= runAll()
    time, accuracy =  [50*(element-time[-1]*0.78) for element in time], [(100 - element)/100 for element in accuracy]
    time2, accuracy2 =  [50*(element-time2[-1]*0.78) for element in time2], [(100 - element)/100 for element in accuracy2]
    #print(time,accuracy)
    time[0] = time[0]/time[-2]*time[-1]
    time2[0] = time2[0]/time2[-2]*time2[-1]
    #print(time, accuracy, time2, accuracy2)
    ammE = []
    ammE2 = []
    algosVec=['int8', 'crs', 'countSketch', 'cooFD', 'blockLRA', 'fastjlt', 'rip', 'smp-pca', 'weighted-cr', 'tugOfWar', 'int8_fp32', 'mm']
    for tag in algosVec:
        old_stdout = sys.stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        try:
            runOnce(a = tag, e = 0)
        except:
            pass
        sys.stdout = old_stdout

        # Get the captured output
        output_before_error = captured_output.getvalue()
        lines = output_before_error.split('\n')

        # Iterate over each line
        for line in lines:
            # Check if A is in the current line
            if tag in line:
                ammE.append(float(line.split(" ")[1]))
    for tag in algosVec:
        old_stdout = sys.stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        try:
            runOnce(a = tag, h = 2000, e = 0)
        except:
            pass
        sys.stdout = old_stdout

        # Get the captured output
        output_before_error = captured_output.getvalue()
        lines = output_before_error.split('\n')

        # Iterate over each line
        for line in lines:
            # Check if A is in the current line
            if tag in line:
                ammE2.append(float(line.split(" ")[1]))
    # PQ, VQ
    ammE.insert(6, 0.0)
    ammE.insert(7, 0.0)
    ammE2.insert(6, 0.0)
    ammE2.insert(7, 0.0)
    return np.array(time), np.array(accuracy), np.array(ammE), np.array(time2), np.array(accuracy2), np.array(ammE2)
main()
