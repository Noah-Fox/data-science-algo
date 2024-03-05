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

    #use three groups of starting medoids
    startingMedoids = [['F15B5', 'F15F3', 'F11D4'],['F6A4', 'F9A2', 'F7F3'],['F12B2', 'F7F3', 'F16F4']]#found testing 10000 combinations
    clusteringTitles = ['similarity','distance','balance']
    clusteringDescs = [
        'Highest intra-cluster similarity',
        'Highest inter-cluster distance',
        'Balance between intra-cluster similarity and inter-cluster distance'
    ]

    #process each medoid group
    for medoids,title,desc in zip(startingMedoids,clusteringTitles,clusteringDescs):
        #run k-medoids clustering
        clusters,clusterMedoids = h1_mod.runKMedoidsClustering(medoids,hist1NPs,npJaccards)
        clusterScore = h1_mod.assignClusteringScores(clusters,clusterMedoids,npJaccards)['similarityAvg']

        #For each cluster, find the average percentage of windows with each feature across each NP
        features = ['Hist1','Vmn','LAD','RNAPII-S2P','RNAPII-S5P','RNAPII-S7P','Enhancer','H3K9me3','H3K20me3','h3k27me3','H3K36me3','NANOG','pou5f1','sox2','CTCF-7BWU']
        clusterFeaturePercentages = [{f: 0 for f in features} for c in clusters]
        for i,c in zip(clusterFeaturePercentages,clusters):
            for f in features:
                npPercentages = [100*sum(hist1WindowDetectionsDf[np] & featureDf[f])/sum(hist1WindowDetectionsDf[np]) for np in c]
                i[f] = sum(npPercentages)/len(c)

        #Find the percentage of hist1 genes with each feature
        histFeaturePercentages = {f:100*featureDf[f].astype(bool).sum()/len(list(featureDf[f])) for f in features}

        #Make a radar chart
        fig = go.Figure()

        for i,(cfp,med) in enumerate(zip(clusterFeaturePercentages,clusterMedoids)):
            df = pd.DataFrame([[f,cfp[f]] for f in features])
            fig.add_trace(go.Scatterpolar(
                r=df[1],
                theta=df[0],
                fill='toself',
                name='Cluster ' + str(i) + ': ' + med,
            ))
        df = pd.DataFrame([[f,histFeaturePercentages[f]] for f in features])
        fig.add_trace(go.Scatterpolar(
            r=df[1],
            theta=df[0],
            name='Hist1 Full Average',
            mode='markers'
        ))
        
        #output to file
        fig.write_image(file='charts/feature-selection-4/' + title + '-radar.png',format='png')
        fig.write_html(file='charts/feature-selection-4/' + title + '-radar.html')
        if input('Open radar diagram (y/n)? ') == 'y':
            fig.show()
        outputFile.write('## ' + desc + '\n\n')
        outputFile.write('![Radar graph](../charts/feature-selection-4/' + title + '-radar.png)\n\n')
        outputFile.write('Cluster medoids: ' + str(clusterMedoids) + '\n\n')
        outputFile.write('Average similarity of each NP to its medoid: ' + str(clusterScore) + '\n\n')
        outputFile.write('Size of each cluster: ' + str([len(c) for c in clusters]) + '\n\n')


    return 

def clusterDistanceScore(clusters,clusterMedoids,npJaccards):
    distances = sum([sum([sum([(1-npJaccards[np][clusterMedoids[(i+x)%3]]) for x in [1,2]]) for np in c]) for i,c in enumerate(clusters)])
    return distances/(2*sum([len(c) for c in clusters]))

if __name__ == "__main__":
    main()
