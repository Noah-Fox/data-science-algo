import numpy
import math
import pandas as pd
import matplotlib.pyplot as plt
import hist1_analysis as h1_mod
import seaborn as sns
import random

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

#Given lists of NPs in each cluster and the medoid of each cluster, return the sum of distances from every NP to its medoid
def assessClusteringQuality(clusters, medoids, npJaccards):
    return sum([sum([(1-npJaccards[np][med]) for np in c]) for (c, med) in zip(clusters,medoids)])

#Assign each element of hist1NPs to a cluster with an element of clusterMedoids at its center
#Each element of the returned list denotes an NP, containing [cluster, np]
def assignKMeansCluster(hist1NPs, clusterMedoids, npJaccards, preferences):
    return [[assignCluster(clusterMedoids,n,npJaccards,pref),n] for (n,pref) in zip(hist1NPs,preferences)]

#Given an NP and the NPs at the center of each cluster, determine which cluster is closest
def assignCluster(clusterMedoids, np, npJaccards, preference):
    clusterSimilarities = [npJaccards[np][c] for c in clusterMedoids]
    return randMax(clusterSimilarities,randOffset=preference)

#Given the cluster assigned to each element, return k lists of NPs in each cluster
def arrangeClusters(clusterAssignments, k):
    return [list(map(lambda x : x[1], filter(lambda x: x[0] == i, clusterAssignments))) for i in range(0,k)]

#Given a list of NPs in a cluster, find the medoid - the NP with the maximum average similarity to all other NPs in the cluster
def findClusterMedoid(cluster, npJaccards):
    avgSimilarities = [sum([(npJaccards[n][cn]) for cn in cluster])/len(cluster) for n in cluster]
    return cluster[avgSimilarities.index(max(avgSimilarities))]

#Given a list of numbers, return the index of the maximum number
#If there are multiple maximums, choose one randomly
def randMax(dataList,randOffset=-1):
    maxVal = max(dataList)
    doubledList = dataList + dataList
    if (randOffset == -1):
        randOffset = random.randint(0,len(dataList)-1)
    return doubledList.index(maxVal,randOffset) % len(dataList)

#Given two NPs, find their Normalized Jaccard Index - Jn(A,B) = (A INTERSECTION B) / min(|A|, |B|)
def normalizedJaccard(npA, npB, hist1WindowDetectionsDf):
    A = (hist1WindowDetectionsDf[npA] == 1)
    B = (hist1WindowDetectionsDf[npB] == 1)
    if (A | B).sum() == 0: return 0
    return (A & B).sum() / min(A.sum(), B.sum())
