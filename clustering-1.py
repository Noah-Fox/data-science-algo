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

    #open output file
    outputFile = open("reports/clustering-1-report.md", 'w')
    outputFile.write('# Clustering 1 Report\n')

    #columns denote an NP, rows denote a window
    windowDetectionsDf = df.iloc[:,3:]

    #columns denote chromosome name and start and stop position, rows denote a window
    windowValuesDf = df.iloc[:,0:3]

    #extract Hist1 windows and NPs
    hist1Windows = h1_mod.findHist1Windows(windowValuesDf)
    hist1WindowDetectionsDf = windowDetectionsDf.iloc[hist1Windows,:]
    hist1NpSums = h1_mod.windowsPerNP(hist1WindowDetectionsDf)
    hist1NPs = [x for x in hist1NpSums.index]

    #construct a matrix of Jaccard Indices for each pair of NPs
    jaccardData = [[normalizedJaccard(a,b,hist1WindowDetectionsDf) for a in hist1NPs] for b in hist1NPs]
    npJaccards = pd.DataFrame(data=jaccardData, index=hist1NPs, columns=hist1NPs)
    
    #make a heat map for similarities
    plt.figure()
    sns.heatmap(npJaccards,cmap='Blues')
    plt.title('Normalized Jaccard Similarity Heat Map for Hist1 NPs')
    plt.savefig('charts/activity-5-similarity-heat-map.png')
    outputFile.write('![Jaccard Similarity Heat Map](../charts/activity-5-similarity-heat-map.png)\n\n')

    #Use k-means clustering to form 3 clusters of NPs
    clusterNPs = random.sample(hist1NPs,3)
    clusterAssignments = assignKMeansCluster(hist1NPs, clusterNPs, npJaccards)
    clusters = arrangeClusters(clusterAssignments, 3)
    running = True
    iterations = 1
    
    while (running):
        clusterMedoids = [findClusterMedoid(c,npJaccards) for c in clusters]
        nextAssignments = assignKMeansCluster(hist1NPs, clusterMedoids, npJaccards)
        nextClusters = arrangeClusters(clusterAssignments,3)
        if (nextAssignments == clusterAssignments):
            running = False
        clusterAssignments = nextAssignments
        clusters = nextClusters
        iterations += 1
    
    outputFile.write('Clusters were found after ' + str(iterations) + ' iterations\n\n')
    #make a heat map for each cluster
    for i,c in enumerate(clusters):
        clusterMatrix = [[(npJaccards[npA][npB] if (npA in c and npB in c and npA != npB) else 0) for npA in hist1NPs] for npB in hist1NPs]
        clusterDf = pd.DataFrame(data=clusterMatrix,index=hist1NPs,columns=hist1NPs)
        plt.figure()
        sns.heatmap(clusterDf,cmap='Greens')
        plt.title('Cluster ' + str(i) + ' Heat Map')
        plotFileName = 'charts/clustering-1-cluster-' + str(i) + 'heat-map.png'
        plt.savefig(plotFileName)
        outputFile.write('![Cluster ' + str(i) + ' Heat Map](../' + plotFileName + ')\n\n')
        outputFile.write('Cluster ' + str(i) + ' contains ' + str(len(c)) + ' elements\n\n')


    outputFile.close()
    return

#Assign each element of hist1NPs to a cluster with an element of clusterNPs at its center
def assignKMeansCluster(hist1NPs, clusterNPs, npJaccards):
    return [[assignCluster(clusterNPs,n,npJaccards),n] for n in hist1NPs]

#Given the cluster assigned to each element, return k lists of NPs in each cluster
def arrangeClusters(clusterAssignments, k):
    return [list(map(lambda x : x[1], filter(lambda x: x[0] == i, clusterAssignments))) for i in range(0,k)]

#Given a list of NPs in a cluster, find the medoid - the NP with the maximum average similarity to all other NPs in the cluster
def findClusterMedoid(cluster, npJaccards):
    avgSimilarities = [sum([(npJaccards[n][cn]) for cn in cluster])/len(cluster) for n in cluster]
    return cluster[randMax(avgSimilarities)]

#Given an NP and the NPs at the center of each cluster, determine which cluster is closest
def assignCluster(clusterNPs, np, npJaccards):
    clusterSimilarities = [npJaccards[np][c] for c in clusterNPs]
    return randMax(clusterSimilarities)

#Given a list of numbers, return the index of the maximum number
#If there are multiple maximums, choose one randomly
def randMax(dataList):
    maxVal = max(dataList)
    doubledList = dataList + dataList
    return doubledList.index(maxVal,random.randint(0,len(dataList)-1)) % len(dataList)

#Given two NPs, find their Normalized Jaccard Index - Jn(A,B) = (A INTERSECTION B) / min(|A|, |B|)
def normalizedJaccard(npA, npB, hist1WindowDetectionsDf):
    A = (hist1WindowDetectionsDf[npA] == 1)
    B = (hist1WindowDetectionsDf[npB] == 1)
    if (A | B).sum() == 0: return 0
    return (A & B).sum() / min(A.sum(), B.sum())


if __name__ == "__main__":
    main()
