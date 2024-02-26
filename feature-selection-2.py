import numpy
import math
import pandas as pd
import matplotlib.pyplot as plt
import hist1_analysis as h1_mod
import seaborn as sns
import random

def main():
    #input data file
    dataFile = 'GSE64881_segmentation_at_30000bp.passqc.multibam.txt'
    df = pd.read_csv(dataFile,sep='\t')

    featureFile = 'Hist1_region_features.csv'
    featureDf = pd.read_csv(featureFile)

    #open output file
    outputFile = open("reports/feature-selection-2-report.md", 'w')
    outputFile.write('# Feature Selection 2 Report\n')

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

    #construct a matrix of Jaccard Indices for each pair of NPs
    jaccardData = [[h1_mod.normalizedJaccard(a,b,hist1WindowDetectionsDf) for a in hist1NPs] for b in hist1NPs]
    npJaccards = pd.DataFrame(data=jaccardData, index=hist1NPs, columns=hist1NPs)

    #run k-means clustering
    medoids = ['F11C2', 'F6A4', 'F7F3']#found testing 10000 combinations
    clusters,clusterMedoids = h1_mod.runKMedoidsClustering(medoids,hist1NPs,npJaccards)

    #determine for each cluster the percentage of windows in each NP that contain histone genes
    clusterHistPercentages = [[100*sum(hist1WindowDetectionsDf[np] & featureDf['Hist1'])/sum(hist1WindowDetectionsDf[np]) for np in c] for c in clusters]
    plt.figure()
    sns.boxplot(data=clusterHistPercentages)
    sns.stripplot(data=clusterHistPercentages,color='Black')
    plt.title('Percentage of histone genes in each NP')
    saveToFile = 'charts/feature-selection-2/cluster-histone-boxplot.png'
    plt.savefig(saveToFile)
    outputFile.write('![Percentage of histone genes in cluster NPs](../' + saveToFile + ')\n\n')

    #determine for each cluster the percentage of windows in each NP that contain LADs
    clusterLadPercentages = [[100*sum(hist1WindowDetectionsDf[np] & featureDf['LAD'])/sum(hist1WindowDetectionsDf[np]) for np in c] for c in clusters]
    plt.figure()
    sns.boxplot(data=clusterLadPercentages)
    sns.stripplot(data=clusterLadPercentages,color='Black')
    plt.title('Percentage of LAPs in each NP')
    saveToFile = 'charts/feature-selection-2/cluster-LAD-boxplot.png'
    plt.savefig(saveToFile)
    outputFile.write('![Percentage of LADs in cluster NPs](../' + saveToFile + ')\n\n')

    return 

if __name__ == "__main__":
    main()
