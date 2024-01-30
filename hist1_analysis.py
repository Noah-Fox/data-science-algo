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

#Returns a series of all NPs that detect hist1 windows, and the amount of windows they detect
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
