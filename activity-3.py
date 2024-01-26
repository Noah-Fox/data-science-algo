import numpy
import math
import pandas as pd
import matplotlib.pyplot as plt

def main():
    #input data file
    dataFile = 'GSE64881_segmentation_at_30000bp.passqc.multibam.txt'
    df = pd.read_csv(dataFile,sep='\t')

    #open output file
    outputFile = open("activity-3-report.md", 'w')
    outputFile.write("# Activity 3 Report\n")
    outputFile.write("The Hist1 region is present on mouse chromosome 13 between 21.7 and 24.1 Mb\n")

    #columns denote an NP, rows denote a window
    windowDetectionsDf = df.iloc[:,3:]

    #columns denote chromosome name and start and stop position, rows denote a window
    windowValuesDf = df.iloc[:,0:3]

    #Step 1
    hist1Windows = findHist1Windows(windowValuesDf)
    hist1WindowDetectionsDf = windowDetectionsDf.iloc[hist1Windows,:]
    outputFile.write("### 1. Number of genomic windows\n")
    outputFile.write("The Hist1 gene appears on " + str(len(hist1Windows)) + " windows\n")

    #Step 2
    hist1NpSums = windowsPerNP(hist1WindowDetectionsDf)
    outputFile.write("### 2. Number of NPs\n")
    outputFile.write("The Hist1 gene is detected by " + str(len(hist1NpSums)) + " NPs\n")
    
    #Step 3
    npDetectionAvg = hist1NpSums.mean()
    outputFile.write("### 3. Average windows detected by an NP\n")
    outputFile.write("An average of " + str(npDetectionAvg) + " Hist1 windows are detected by Hist1 NPs\n")
    
    #Step 4
    npDetectionMin = hist1NpSums.min()
    npDetectionMax = hist1NpSums.max()
    outputFile.write("### 4. Range of windows detected by an NP\n")
    outputFile.write("The minimum amount of Hist1 windows detected by an NP is " + str(npDetectionMin) + "\n\n")
    outputFile.write("The maximum amount of Hist1 windows detected by an NP is " + str(npDetectionMax) + "\n")

    #Step 5
    hist1WindowDetections = NPsPerWindow(hist1WindowDetectionsDf)
    windowDetectionAvg = hist1WindowDetections.mean()
    windowDetectionMin = hist1WindowDetections.min()
    windowDetectionMax = hist1WindowDetections.max()
    outputFile.write("### 5. Detections per window\n")
    outputFile.write("The minimum amount of NPs that detect a Hist1 window is " + str(windowDetectionMin) + "\n\n")
    outputFile.write("The maximum amount of detections of a Hist1 window is " + str(windowDetectionMax) + "\n\n")
    outputFile.write("The average amount of detections is " + str(windowDetectionAvg) + "\n")

    #Step 6
    npRadialPositions = findNpRadialPositions(windowDetectionsDf)
    hist1Nps = list(hist1NpSums.keys())
    hist1NpRadialPositions = list(map(lambda x: npRadialPositions[x], hist1Nps))
    radialPositionCounts = list(map(lambda x: hist1NpRadialPositions.count(x), [1,2,3,4,5]))
    radialPositionMode = radialPositionCounts.index(max(radialPositionCounts))+1
    outputFile.write("### 6. Radial positions of Hist1 Genes\n")
    outputFile.write("![Radial Position Bar Graph](./activity-3-radial-positions.png)\n\n")
    outputFile.write("The most commonly occurring radial position of a Hist1 window is " + str(radialPositionMode) + "\n")

    plt.figure()
    plt.bar([1,2,3,4,5],radialPositionCounts)
    plt.xlabel('Radial Position (Apical - Equitorial)')
    plt.ylabel('Occurrence of radial position')
    plt.savefig('activity-3-radial-positions.png')
    
    #step 7
    windowCompactions = findWindowCompactions(windowDetectionsDf)
    hist1WindowCompactions = list(map(lambda x: windowCompactions[x], hist1Windows))
    windowCompactionCounts = list(map(lambda x: hist1WindowCompactions.count(x), [1,2,3,4,5,6,7,8,9,10]))
    outputFile.write("### 7. Compactions of Hist1 Windows\n")
    outputFile.write("![Window compaction Bar Graph](./activity-3-window-compactions.png)\n\n")
    outputFile.write(".\n")

    plt.figure()
    plt.bar([1,2,3,4,5,6,7,8,9,10],windowCompactionCounts)
    plt.xlabel('Compaction of a window (most condensed - least condensed)')
    plt.ylabel('Occurence of compaction')
    plt.savefig('activity-3-window-compactions.png')
    

    outputFile.close()
    return 


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

#for each window, rate its compaction between 1 (most condensed) and 10 (least condensed)
def findWindowCompactions(windowDetectionsDf):
    #sorts into ten equal groups -- wrong way?
    windowSumsSorted = windowDetectionsDf.sum(axis=1).sort_values()
    for index,key in enumerate(windowSumsSorted.keys()):
        windowSumsSorted[key] = math.floor(index / (len(windowSumsSorted)/10)) + 1
    return windowSumsSorted


if __name__ == "__main__":
    main()
