import numpy
import math
import pandas as pd
import matplotlib.pyplot as plt
import hist1_analysis as h1_mod
import seaborn as sns


def main():
    #input data file
    dataFile = 'GSE64881_segmentation_at_30000bp.passqc.multibam.txt'
    df = pd.read_csv(dataFile,sep='\t')

    #open output file
    outputFile = open("reports/activity-5-report.md", 'w')
    outputFile.write('# Activity 5 Report\n')
    outputFile.write('The Jaccard Index of two objects, A and B, with n binary attributes, determines the similarity of the objects\n\n')
    outputFile.write('The index/similarity J(A, B) can be found with the equation `|A INTERSECT B| / |A UNION B|`\n\n')
    outputFile.write('The Jaccard Distance can be found 1-J\n\n')

    #columns denote an NP, rows denote a window
    windowDetectionsDf = df.iloc[:,3:]

    #columns denote chromosome name and start and stop position, rows denote a window
    windowValuesDf = df.iloc[:,0:3]

    #extract Hist1 windows and NPs
    hist1Windows = h1_mod.findHist1Windows(windowValuesDf)
    hist1WindowDetectionsDf = windowDetectionsDf.iloc[hist1Windows,:]
    hist1NpSums = h1_mod.windowsPerNP(hist1WindowDetectionsDf)
    hist1NPs = hist1NpSums.index

    #construct a matrix of Jaccard Indices for each pair of NPs
    jaccardData = [[jaccard(a,b,hist1WindowDetectionsDf) for a in hist1NPs] for b in hist1NPs]
    npJaccards = pd.DataFrame(data=jaccardData, index=hist1NPs, columns=hist1NPs)

    #construct a matrix of Jaccard Distances for each pair of NPs
    jaccardDistData = [[1-jaccard(a,b,hist1WindowDetectionsDf) for a in hist1NPs] for b in hist1NPs]
    npJaccardDists = pd.DataFrame(data=jaccardDistData, index=hist1NPs, columns=hist1NPs)
    
    #make a heat map for similarities
    plt.figure()
    sns.heatmap(npJaccards,cmap='Blues')
    plt.title('Jaccard Similarity Heat Map for Hist1 NPs')
    plt.savefig('charts/activity-5-similarity-heat-map.png')
    outputFile.write('![Jaccard Similarity Heat Map](../charts/activity-5-similarity-heat-map.png)\n')

    #make a heat map for the amount of detections by each NP
    detectionAmountData = [[hist1WindowDetectionsDf[a].sum() for a in hist1NPs] for b in hist1NPs]
    npDetectionAmount = pd.DataFrame(data=detectionAmountData, index=hist1NPs, columns=hist1NPs)
    plt.figure()
    sns.heatmap(npDetectionAmount,cmap='Blues')
    plt.title('NP Detection Amounts')
    plt.savefig('charts/activity-5-np-detection-amount-heat-map.png')
    outputFile.write('![NP Detection Amount Heat Map](../charts/activity-5-np-detection-amount-heat-map.png)\n\n')
    outputFile.write('''Rows and columns are visible in the Jaccard Similarity heat map above. 
                        As can be seen in the NP Detection Amounts chart, these patterns correlate directly to the amount of 
                        windows detected by the NP\n\n''')
    
    simValData = [[jaccard(a,b,hist1WindowDetectionsDf)/hist1WindowDetectionsDf[a].sum() for a in hist1NPs] for b in hist1NPs]
    simValMatrix = pd.DataFrame(data=simValData,index=hist1NPs,columns=hist1NPs)
    plt.figure()
    sns.heatmap(simValMatrix, cmap = "Blues")
    plt.savefig('charts/activity-5-similarity-values.png')
    outputFile.write('![Jaccard Similarity Values](../charts/activity-5-similarity-values.png)\n\n')
    outputFile.write('The Jaccard Similarity heat map can be clarified by dividing the similarity by the detection count\n\n')

    #make a heat map for distances
    plt.figure()
    sns.heatmap(npJaccardDists,cmap='Greens')
    plt.title('Jaccard Distance Heat Map for Hist1 NPs')
    plt.savefig('charts/activity-5-distance-heat-map.png')
    outputFile.write('![Jaccard Distance Heat Map](../charts/activity-5-distance-heat-map.png)\n')

    


    outputFile.close()
    return



#Given two NPs, find their Jaccard Index
def jaccard(npA, npB, hist1WindowDetectionsDf):
    A = (hist1WindowDetectionsDf[npA] == 1)
    B = (hist1WindowDetectionsDf[npB] == 1)
    if (A & B).sum() == 0: return 0
    return  (A & B).sum() / (A | B).sum()

if __name__ == "__main__":
    main()
