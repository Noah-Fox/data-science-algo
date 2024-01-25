import numpy
import math
import pandas as pd
import matplotlib.pyplot as plt

def main():
    #input data file
    dataFile = 'GSE64881_segmentation_at_30000bp.passqc.multibam.txt'
    df = pd.read_csv(dataFile,sep='\t')

    #columns denote an NP, rows denote a window
    windowDetectionsDf = df.iloc[:,3:]

    #columns denote chromosome name and start and stop position, rows denote a window
    windowValuesDf = df.iloc[:,0:3]

    #Step 1
    hist1Windows = findHist1Windows(windowValuesDf)
    hist1WindowDetectionsDf = windowDetectionsDf.iloc[hist1Windows,:]

    #Step 2
    hist1NpSums = windowsPerNP(hist1WindowDetectionsDf)
    
    #Step 3
    npDetectionAvg = hist1NpSums.mean()
    
    #Step 4
    npDetectionMin = hist1NpSums.min()
    npDetectionMax = hist1NpSums.max()

    #Step 5
    hist1WindowDetections = NPsPerWindow(hist1WindowDetectionsDf)
    windowDetectionAvg = hist1WindowDetections.mean()
    windowDetectionMin = hist1WindowDetections.min()
    windowDetectionMax = hist1WindowDetections.max()

    #Step 6
    npRadialPositions = findNpRadialPositions(windowDetectionsDf)
    print(npRadialPositions)


#returns a list of numbers of chr13 windows that are between 21.7 and 24.1 Mb
def findHist1Windows(windowValuesDf):
    chromType = "chr13"
    startRange = 21700000
    endRange = 24100000
    hist1Windows = []

    for index in range(0,windowValuesDf.shape[0]):
        windowStart = windowValuesDf['start'][index]
        windowEnd = windowValuesDf['stop'][index]
        windowChrom = windowValuesDf['chrom'][index]
        if (windowStart >= startRange and windowEnd <= endRange and windowChrom == chromType):
            hist1Windows.append(index)
    return hist1Windows

#for each NP in hist1NPs, finds the amount of hist1Windows detected by it. Returns series
def windowsPerNP(hist1WindowDetectionsDf):
    return hist1WindowDetectionsDf.sum(axis=0).where(lambda x : x != 0).dropna()

#for each window in hist1Windows, finds the amount of hist1NPs that detects it. Returns series
def NPsPerWindow(hist1WindowDetectionsDf):
    return hist1WindowDetectionsDf.sum(axis=1)

#for each np, rate its radial position between 1 (apical) and 5 (equitorial)
def findNpRadialPositions(windowDetectionsDf):
    npSumsSorted = windowDetectionsDf.sum(axis=0).sort_values()
    for index,key in enumerate(npSumsSorted.keys()):
        npSumsSorted[key] = math.floor(index / (len(npSumsSorted)/5)) + 1
    return npSumsSorted

#TODO: radial positions of windows, compactions of windows (fix script)

if __name__ == "__main__":
    main()
