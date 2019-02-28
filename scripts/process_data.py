import numpy as np
import pandas as pd
import argparse
from collections import Counter
from argparse import ArgumentParser
from Preprocessor import Preprocessor

def read_file(filename):
    data = pd.read_csv(filename, sep='\t', header=None, usecols=[2,3])
    return data

def process_data(data):
    neutralIndices = []
    emptyIndices = []
    labels = data.iloc[:,0].tolist()
    for i in range(len(labels)):
        if labels[i] == -2:
            data.at[i,2] = -1
        elif labels[i] == 2:
            data.at[i,2] = 1
        #elif labels[i] == 0:
        #    neutralIndices.append(i)
    #processed = data.drop(neutralIndices)
    processed = data
    #upsample class
    multiplier = upsample_multiplier(processed, -1, 1)
    isNeg = processed[2] == -1
    df_try = processed[isNeg]
    data2 = processed.append([df_try]*multiplier,ignore_index=True)

    multiplier = upsample_multiplier(processed, 0, 1)
    isNeut = processed[2] == 0
    df_try = processed[isNeut]
    data2 = data2.append([df_try]*multiplier,ignore_index=True)

    labels2 = data2.iloc[:,0].tolist()
    #for i in range(len(labels2)):
    #    if labels2[i] == -1:
    #        data2.at[i,2] = 0

    p = Preprocessor()
    texts = data2.iloc[:,1].tolist()
    processed_texts = p.preprocess(texts)
    labels = data2.iloc[:,0].tolist()
    write_csv(processed_texts, labels)

def upsample_multiplier(data, value, high):
     labels = data.iloc[:,0].tolist()
     counts = Counter()

     for label in labels:
         counts[label] += 1

     multiplier = 1
     while counts[value]*multiplier < counts[high]:
        multiplier += 1

     return multiplier

def write_csv(texts, labels):
    numSamples = len(texts)
    with open(args.o, 'w') as f:
        for i in range(numSamples):
            f.write(texts[i])
            f.write(',')
            f.write(str(labels[i]))
            if i < numSamples - 1:
                f.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='Input filename', required=True)
    parser.add_argument('-o', help='Output filename', required=True)
    args = parser.parse_args()
    data = read_file(args.i)
    data = process_data(data)
