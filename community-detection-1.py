import numpy
import math
import pandas as pd
import matplotlib.pyplot as plt
import hist1_analysis as h1_mod
import seaborn as sns
import random
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

def main():
    #input data file
    dataFile = 'GSE64881_segmentation_at_30000bp.passqc.multibam.txt'
    df = pd.read_csv(dataFile,sep='\t')

    featureFile = 'Hist1_region_features.csv'
    featureDf = pd.read_csv(featureFile)

    #open output file
    outputFile = open("reports/community-detection-1-report.md", 'w')
    outputFile.write('# Community Detection Report\n')

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
    for i in indices:
        for x in indices:
            if i != x:
                sum += linkageTable.loc[i,x]
                div += 1
    averageLinkage = sum/div

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

    #find degree centrality of each window, and the hubs (top 5)
    degreeCentrality = (linkageGraph.sum() / (len(indices)-1)).sort_values(ascending=False)
    hubs = list(degreeCentrality.head(5).index)
    
    #report data on each community
    for c,h in enumerate(hubs):
        community = []
        for i in indices:
            if linkageGraph.loc[h,i] or i == h:
                community.append(i)
        hist1Count = 0
        ladCount = 0
        for i in community:
            if featureDf.loc[i,'Hist1'] == 1:
                hist1Count += 1
            if featureDf.loc[i,'LAD'] == 1:
                ladCount += 1
        outputFile.write(f'### Community {c+1}\n\n')
        outputFile.write(f'Size: {len(community)} nodes\n\n')
        outputFile.write(f'{hist1Count} nodes contain hist1 gene\n\n')
        outputFile.write(f'{ladCount} nodes contain a LAD\n\n')

        outputFile.write(f'Nodes in community:\n\n```')
        for i in community:
            outputFile.write(f'{i} ')
        outputFile.write('```\n\n')

        #create graph of community
        g = nx.Graph()
        for i in community:
            for j in community:
                if i > j and linkageGraph.loc[i,j]:
                    g.add_edge(i,j)
        pos = nx.spring_layout(g,iterations=1000)
        plt.figure()
        for i in community:
            nx.draw_networkx_nodes(g,nodelist=[i],node_size=(degreeCentrality[i]*99+1),pos=pos)
        nx.draw_networkx_edges(g,pos=pos)

        plt.tight_layout()
        savefile = f'charts/community-detection-1/community-{c+1}-graph.png'
        plt.savefig(savefile)
        outputFile.write(f'![Community {c+1}](../{savefile})\n\n')

    


    return 


if __name__ == "__main__":
    main()
