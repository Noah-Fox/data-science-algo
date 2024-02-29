import numpy
import math
import pandas as pd
import matplotlib.pyplot as plt
import hist1_analysis as h1_mod
import seaborn as sns
import random
import plotly.express as px

def main():
    #input data file
    dataFile = 'GSE64881_segmentation_at_30000bp.passqc.multibam.txt'
    df = pd.read_csv(dataFile,sep='\t')

    featureFile = 'Hist1_region_features.csv'
    featureDf = pd.read_csv(featureFile)

    #open output file
    outputFile = open("reports/feature-selection-4-report.md", 'w')
    outputFile.write('# Feature Selection 4 Report\n')

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

    #For each cluster, find the average percentage of windows with each feature across each NP
    features = ['Hist1','Vmn','LAD','RNAPII-S2P','RNAPII-S5P','RNAPII-S7P','Enhancer','H3K9me3','H3K20me3','h3k27me3','H3K36me3','NANOG','pou5f1','sox2','CTCF-7BWU']
    clusterFeaturePercentages = [{f:sum([100*sum(hist1WindowDetectionsDf[np] & featureDf[f])/sum(hist1WindowDetectionsDf[np]) for np in c])/len(c) for f in features} for c in clusters]
    clusterFeaturePercentages = [{f: 0 for f in features} for c in clusters]
    for i,c in zip(clusterFeaturePercentages,clusters):
        for f in features:
            npPercentages = [100*sum(hist1WindowDetectionsDf[np] & featureDf[f])/sum(hist1WindowDetectionsDf[np]) for np in c]
            i[f] = sum(npPercentages)/len(c)

    #Make a radar chart
    df = pd.DataFrame([[f,clusterFeaturePercentages[0][f]] for f in features])
    print(df)
    fig = px.line_polar(df,r=1,theta=0,line_close=True)
    fig.show()

    return 

if __name__ == "__main__":
    main()
