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
    outputFile = open("reports/feature-selection-3-report.md", 'w')
    outputFile.write('# Feature Selection 3 Report\n')

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

    #Find radial positions of all NPs
    npRadialPositions = h1_mod.findNpRadialPositions(windowDetectionsDf)

    #construct a matrix of Jaccard Indices for each pair of NPs
    jaccardData = [[h1_mod.normalizedJaccard(a,b,hist1WindowDetectionsDf) for a in hist1NPs] for b in hist1NPs]
    npJaccards = pd.DataFrame(data=jaccardData, index=hist1NPs, columns=hist1NPs)

    #run k-means clustering
    startingMedoids = [['F15B5', 'F15F3', 'F11D4'],['F6A4', 'F9A2', 'F7F3'],['F12B2', 'F7F3', 'F16F4']]#similarity,distance,balance
    medoids = startingMedoids[0]
    clusters,clusterMedoids = h1_mod.runKMedoidsClustering(medoids,hist1NPs,npJaccards)

    #for each cluster, find the amount of NPs with each radial position
    radialPositionCounts = [{1:0, 2:0, 3:0, 4:0, 5:0} for x in [0,1,2]]
    for c, count in zip(clusters,radialPositionCounts):
        for np in c:
            count[npRadialPositions[np]] += 1

    for i in [0,1,2]:
        plt.figure()
        plt.bar([1,2,3,4,5],list(radialPositionCounts[i].values()))
        plt.title('Cluster ' + str(i) + ' NP radial positions')
        plt.xlabel('Radial positions (Apical - Equitorial)')
        plt.ylabel('NPs in each radial position')
        saveFile = 'charts/feature-selection-3/cluster-' + str(i) + '-radial-counts.png'
        plt.savefig(saveFile)
        outputFile.write('![Cluster ' + str(i) + ' radial positions](../' + saveFile + ')\n\n')


    return 

if __name__ == "__main__":
    main()
