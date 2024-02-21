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
    outputFile = open("reports/feature-selection-report.md", 'w')
    outputFile.write('# Feature Selection Report\n')

    #columns denote an NP, rows denote a window
    windowDetectionsDf = df.iloc[:,3:]

    #columns denote chromosome name and start and stop position, rows denote a window
    windowValuesDf = df.iloc[:,0:3]

    #extract Hist1 windows and NPs
    hist1Windows = h1_mod.findHist1Windows(windowValuesDf)
    hist1WindowDetectionsDf = windowDetectionsDf.iloc[hist1Windows,:]
    hist1NpSums = h1_mod.windowsPerNP(hist1WindowDetectionsDf)
    hist1NPs = list(hist1NpSums.index)

    #construct a matrix of Jaccard Indices for each pair of NPs
    jaccardData = [[h1_mod.normalizedJaccard(a,b,hist1WindowDetectionsDf) for a in hist1NPs] for b in hist1NPs]
    npJaccards = pd.DataFrame(data=jaccardData, index=hist1NPs, columns=hist1NPs)

    #Run K-medoids clustering
    npCombos = []
    for x in range(0,len(hist1NPs)):
        for y in range(x+1,len(hist1NPs)):
            for z in range(y+1,len(hist1NPs)):
                npCombos.append([hist1NPs[x],hist1NPs[y],hist1NPs[z]])
    
    clusteringAmount = 100
    selectCombos = random.sample(npCombos,clusteringAmount)
    clusteringScores = []
    maxScore = 0
    optimalClustering = []

    for i in selectCombos:
        print(i,end=' ')
        clusters, clusterMedoids = runKMedoidsClustering(i,hist1NPs,npJaccards)
        clusteringScore = assignClusteringScores(clusters,clusterMedoids,npJaccards)
        clusteringScores.append(clusteringScore)
        print(clusteringScore['similarityAvg'],len(clusteringScores))
        if clusteringScore['similarityAvg'] > maxScore:
            maxScore = clusteringScore['similarityAvg']
            optimalClustering = clusters
    
    plt.figure()
    plt.title('Intra-cluster similarity averages')
    plt.scatter(range(0,len(selectCombos)),list(map(lambda x: x['similarityAvg'], clusteringScores)))
    saveToFile = 'charts/feature-selection/similarity-averages.png'
    plt.savefig(saveToFile)
    outputFile.write('![Intra-cluster similarity averages](../' + saveToFile + ')\n\n')

    outputFile.write(str(clusteringAmount) + ' iterations of k-medoids clustering performed\n\n')
    outputFile.write('Maximum similarity average found: ' + str(maxScore) + '\n\n')


    for i in range(0,3):
        plt.figure()
        sns.heatmap(hist1WindowDetectionsDf.loc[:,optimalClustering[i]],cmap='Blues')
        plt.title('Cluster ' + str(i))
        saveToFile = 'charts/feature-selection/cluster-' + str(i) + '-heatmap.png'
        plt.savefig(saveToFile)
        outputFile.write('![Cluster ' + str(i) + '](../' + saveToFile + ')\n\n')



    return 


def assignClusteringScores(clusters,clusterMedoids,npJaccards):
    npAmount = sum([len(c) for c in clusters])
    scores = {
        'similaritySum': sum([sum([npJaccards[np][med] for np in c]) for c,med in zip(clusters,clusterMedoids)]),
        'similarityAvg': sum([sum([npJaccards[np][med] for np in c]) for c,med in zip(clusters,clusterMedoids)])/npAmount,
        'distanceSum': sum([sum([1-npJaccards[np][med] for np in c]) for c,med in zip(clusters,clusterMedoids)]),
        'distanceAvg': sum([sum([1-npJaccards[np][med] for np in c]) for c,med in zip(clusters,clusterMedoids)])/npAmount
    }
    return scores


#performs K-Medoids clustering on values in hist1NPs, with similarities denoted in npJaccards, starting with clusterMedoids medoids
#returns clusters and clusterMedoids
def runKMedoidsClustering(clusterMedoids, hist1NPs, npJaccards):
    clusterPreferences = [random.randint(0,3) for x in hist1NPs]
        #An arbitrary preference of which cluster to be assigned to in case of a tie
    clusterAssignments = h1_mod.assignKMeansCluster(hist1NPs, clusterMedoids, npJaccards, clusterPreferences)
    clusters = h1_mod.arrangeClusters(clusterAssignments, 3)
    running = True
    iterations = 1
    while (running):
        clusterMedoids = [h1_mod.findClusterMedoid(c,npJaccards) for c in clusters]
        nextAssignments = h1_mod.assignKMeansCluster(hist1NPs, clusterMedoids, npJaccards, clusterPreferences)
        nextClusters = h1_mod.arrangeClusters(clusterAssignments,3)
        if (nextAssignments == clusterAssignments):
            running = False
        clusterAssignments = nextAssignments
        clusters = nextClusters
        iterations += 1

    return clusters, clusterMedoids

if __name__ == "__main__":
    main()
