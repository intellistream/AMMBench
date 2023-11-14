#!/usr/bin/env python3
import os
import torch
import re
import sys
import run as run


def modify_file(file_path, new_string):
    # Pattern to find the line
    pattern = re.compile(r'(return amm\(input, weight, bias, minimal_k, ")(.*?)(")')

    # Read the file contents
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Modify the necessary line
    with open(file_path, 'w') as file:
        for line in lines:
            if pattern.search(line):
                # Replace the content within the quotes
                line = pattern.sub(r'\1' + new_string + r'\3', line)
            file.write(line)


def runAll():
    original_stdout = sys.stdout
    algosVec=['int8', 'crs', 'countSketch', 'cooFD', 'blockLRA', 'fastjlt', 'rip', 'smp-pca', 'weighted-cr', 'tugOfWar', 'int8_fp32', 'mm']
    filePath = os.path.abspath(os.path.join(os.getcwd(), "approx_mul_pytorch/functional")) + "/approx_linear.py"
    for i in range(len(algosVec)):
        algo = algosVec[i]
        resultPath = os.path.abspath(os.path.join(os.getcwd(), "results/")) + "/" + algo
        if os.path.exists(resultPath):
            continue
        #modify_file(filePath, algo)
        print("Running: " + algo)
        with open(resultPath, 'w') as f:
            sys.stdout = f
            run.main(algo)
            sys.stdout = original_stdout

def extract_info(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Regular expression to find the time in the line with "19:"
    time_pattern = r"19：\(wall time: ([\d.]+) sec\)"
    time_match = re.search(time_pattern, content)
    time_19 = time_match.group(1) if time_match else "Not found"

    # Regular expression to find the accuracy in the line with "test acc:"
    accuracy_pattern = r"test acc: ([\d.]+)"
    accuracy_match = re.search(accuracy_pattern, content)
    test_accuracy = accuracy_match.group(1) if accuracy_match else "Not found"

    return time_19, test_accuracy

def parseResult():
    algosVec=['int8', 'crs', 'countSketch', 'cooFD', 'blockLRA', 'fastjlt', 'vq', 'pq', 'rip', 'smp-pca', 'weighted-cr', 'tugOfWar', 'int8_fp32', 'mm']
    accuracy = []
    time = []
    for algo in algosVec:
        resultPath = os.path.abspath(os.path.join(os.getcwd(), "results/")) + "/" + algo
        if os.path.exists(resultPath):
            time_19, test_accuracy = extract_info(resultPath)
            accuracy.append(float(test_accuracy))
            time.append(float(time_19))
        else:
            accuracy.append(100)
            time.append(0)
    return time, accuracy

def main():
    directory_path = "results"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    soSpace = os.path.abspath(os.path.join(os.getcwd(), "../../..")) + "/"
    torch.ops.load_library(soSpace+"libIntelliStream.so")
    runAll()
    time, accuracy = parseResult()
    return time,accuracy


if __name__ == "__main__":
    main()
