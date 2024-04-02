import numpy
import math
import pandas as pd
import matplotlib.pyplot as plt
import hist1_analysis as h1_mod
import seaborn as sns
import random
import plotly.express as px
import plotly.graph_objects as go

def main():
    #input data file
    dataFile = 'GSE64881_segmentation_at_30000bp.passqc.multibam.txt'
    df = pd.read_csv(dataFile,sep='\t')

    featureFile = 'Hist1_region_features.csv'
    featureDf = pd.read_csv(featureFile)

    #open output file
    outputFile = open("reports/network-centrality-1-report.md", 'w')
    outputFile.write('# Network Centrality 1 Report\n')

    #columns denote an NP, rows denote a window
    windowDetectionsDf = df.iloc[:,3:]

    #columns denote chromosome name and start and stop position, rows denote a window
    windowValuesDf = df.iloc[:,0:3]

    #extract Hist1 windows and NPs
    hist1Windows = h1_mod.findHist1Windows(windowValuesDf)
    hist1WindowDetectionsDf = windowDetectionsDf.iloc[hist1Windows,:]
    hist1WindowValuesDf = windowValuesDf.iloc[hist1Windows,:]
    hist1NpSums = h1_mod.windowsPerNP(hist1WindowDetectionsDf)
    hist1NPs = list(hist1NpSums.index)

    #change indices of featureDf to be the same as windowValuesDf
    featureDf = featureDf.set_index(hist1WindowValuesDf.index)

    #create the normalized linkage table
    linkageArray = [[0.0 for i in hist1WindowDetectionsDf.index] for x in hist1WindowDetectionsDf.index]
    indices = hist1WindowDetectionsDf.index
    linkageTable = pd.DataFrame(linkageArray,index=indices,columns=indices)
    for i in indices:
        for x in indices:
            linkageTable.loc[i,x] = h1_mod.normalizedLinkage(hist1WindowDetectionsDf.loc[i,:],hist1WindowDetectionsDf.loc[x,:])

    #find the average of all linkages
    sum = 0
    div = 0
    vals = []
    for i in indices:
        for x in indices:
            if i != x:
                sum += linkageTable.loc[i,x]
                div += 1
                vals.append(linkageTable.loc[i,x])
    averageLinkage = sum/div
    print('average normalized linkage:',averageLinkage)


    #convert linkage table to an adjacency matrix
    linkageGraph = linkageTable.copy(deep=True)
    for i in indices:
        for x in indices:
            if i == x:
                linkageGraph.loc[i,x] = 0 
            elif linkageTable.loc[i,x] > averageLinkage:
                linkageGraph.loc[i,x] = 1
            else:
                linkageGraph.loc[i,x] = 0
    


    return 



if __name__ == "__main__":
    main()
