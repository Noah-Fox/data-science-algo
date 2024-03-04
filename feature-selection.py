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

    maxScore = {'similarityAvg': 0}
    optimalSimClustering = []
    optimalSimMedoids = []

    optimalDistClustering = []
    optimalDistMedoids = []
    maxDist = {'interClusterDistanceAvg': 0}

    balanceScore = {}
    minBalanceDist = 2
    optimalBalanceClustering = []
    optimalBalanceMedoids = []

    for i in selectCombos:
        print(i,end=' ')
        clusters, clusterMedoids = runKMedoidsClustering(i,hist1NPs,npJaccards)
        clusteringScore = assignClusteringScores(clusters,clusterMedoids,npJaccards)
        clusteringScores.append(clusteringScore)
        print(clusteringScore['similarityAvg'],len(clusteringScores))
        if clusteringScore['similarityAvg'] > maxScore['similarityAvg']:
            maxScore = clusteringScore
            optimalSimClustering = clusters
            optimalSimMedoids = clusterMedoids
        if clusteringScore['interClusterDistanceAvg'] > maxDist['interClusterDistanceAvg']:
            maxDist = clusteringScore
            optimalDistClustering = clusters
            optimalDistMedoids = clusterMedoids
        balanceDist = math.sqrt(math.pow(1-clusteringScore['similarityAvg'],2) + math.pow(1-clusteringScore['interClusterDistanceAvg'],2))
        if balanceDist < minBalanceDist:
            minBalanceDist = balanceDist
            balanceScore = clusteringScore
            optimalBalanceClustering = clusters
            optimalBalanceMedoids = clusterMedoids
    
    plt.figure()
    plt.title('Cluster scores')
    plt.xlabel('Inter-cluster distance averages')
    plt.ylabel('Intra-cluster similarity averages')
    plt.scatter(list(map(lambda x: x['interClusterDistanceAvg'], clusteringScores)),list(map(lambda x: x['similarityAvg'], clusteringScores)))
    # plt.scatter(range(0,len(selectCombos)),list(map(lambda x: x['similarityAvg'], clusteringScores)))
    saveToFile = 'charts/feature-selection/similarity-averages.png'
    plt.savefig(saveToFile)
    outputFile.write('![Intra-cluster similarity averages](../' + saveToFile + ')\n\n')

    outputFile.write(str(clusteringAmount) + ' iterations of k-medoids clustering performed\n\n')

    outputFile.write('### Clustering with highest average distance of each NP to every other medoid\n\n')
    outputFile.write('Maximum average distance found: ' + str(maxDist['interClusterDistanceAvg']) + '\n\n')
    outputFile.write('Maximum similarity average found: ' + str(maxDist['similarityAvg']) + '\n\n')
    outputFile.write('Final Medoids: ' + str(optimalDistMedoids) + '\n\n')
    outputFile.write('Size of each cluster: ' + str([len(c) for c in optimalDistClustering]) + '\n\n')

    outputFile.write('### Clustering with highest average similarity of each NP to its medoid\n\n')
    outputFile.write('Maximum similarity average found: ' + str(maxScore['similarityAvg']) + '\n\n')
    outputFile.write('Maximum average distance found: ' + str(maxScore['interClusterDistanceAvg']) + '\n\n')
    outputFile.write('Final Medoids: ' + str(optimalSimMedoids) + '\n\n')
    outputFile.write('Size of each cluster: ' + str([len(c) for c in optimalSimClustering]) + '\n\n')

    outputFile.write('### Clustering with best balance between intra-cluster similarity and inter-cluster distance\n\n')
    outputFile.write('Found by taking the clustering with the minimum distance between (intra-cluster similarity,inter-cluster distance) and (1,1)\n\n')
    outputFile.write('Maximum similarity average found: ' + str(balanceScore['similarityAvg']) + '\n\n')
    outputFile.write('Maximum average distance found: ' + str(balanceScore['interClusterDistanceAvg']) + '\n\n')
    outputFile.write('Final Medoids: ' + str(optimalBalanceMedoids) + '\n\n')
    outputFile.write('Size of each cluster: ' + str([len(c) for c in optimalBalanceClustering]) + '\n\n')

    outputFile.write('## Cluster heat maps for optimized intra-cluster similarity\n\n')
    for i in range(0,3):
        plt.figure()
        sns.heatmap(hist1WindowDetectionsDf.loc[:,optimalSimClustering[i]],cmap='Blues')
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
        'distanceAvg': sum([sum([1-npJaccards[np][med] for np in c]) for c,med in zip(clusters,clusterMedoids)])/npAmount,
        'interClusterDistanceAvg': sum([sum([sum([(1-npJaccards[np][clusterMedoids[(i+x)%3]]) for x in [1,2]]) for np in c]) for i,c in enumerate(clusters)])/(2*npAmount)
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
