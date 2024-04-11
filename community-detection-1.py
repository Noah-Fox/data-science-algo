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

    #create a graph showing all nodes
    g = nx.Graph()
    for i in indices:
        for x in indices:
            if i > x and linkageGraph.loc[i,x]:
                g.add_edge(i,x)
    pos = nx.spring_layout(g,iterations=1000)
    plt.figure()
    for i in indices:
        if degreeCentrality[i] > 0:
            s = 20
            if i in hubs:
                s = 100
            commCount = 0
            for h in hubs:
                if linkageGraph.loc[i,h]:
                    commCount += 1
            colors = ['tab:red','tab:orange','tab:olive','tab:green','tab:blue','tab:purple']
            # colors = [[0,0,255],[0,255,0],[255,0,0],[0,255,255],[255,0,255],[255,255,0]]
            nx.draw_networkx_nodes(g,nodelist=[i],node_size=s,pos=pos,node_color=colors[commCount])
    nx.draw_networkx_edges(g,pos=pos)
    plt.tight_layout()
    savefile = f'charts/community-detection-1/full-network-graph.png'
    plt.savefig(savefile)
    outputFile.write(f'![Full network graph](../{savefile})\n\n')
    outputFile.write(f'This graph shows the full network on windows. The larger nodes are the hubs, with the largest degree centralities. ')
    outputFile.write(f'The nodes are colored from red to purple, according to the number of communities they are in (0-5)\n\n')


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
        outputFile.write(f'{hist1Count} nodes contain hist1 gene: {round(100*hist1Count/len(community),2)}%\n\n')
        outputFile.write(f'{ladCount} nodes contain a LAD: {round(100*ladCount/len(community),2)}%\n\n')

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

        communityTable = [[1 if (i in community and x in community and linkageGraph.loc[i,x]) else 0 for i in indices] for x in indices]
        communityDf = pd.DataFrame(communityTable, index=indices, columns=indices)
        plt.figure()
        sns.heatmap(communityDf,cmap='bwr')
        plt.title(f'Community {c+1}')
        savefile = f'charts/community-detection-1/community-{c+1}-heat-map.png'
        plt.savefig(savefile)
        # outputFile.write(f'![Community {c+1}](../{savefile})\n\n')
    

    outputFile.write(f'## Heat maps\n\nEach heat map shows connections between nodes in a community as red\n\n')
    for c,h in enumerate(hubs):
        outputFile.write(f'![Community {c+1} heat map](../charts/community-detection-1/community-{c+1}-heat-map.png)\n\n')

    


    return 


if __name__ == "__main__":
    main()
