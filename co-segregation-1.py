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
    outputFile = open("reports/co-segregation-1-report.md", 'w')
    outputFile.write('# Co-segregation 1 Report\n')

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
            linkageTable.loc[i,x] = normalizedLinkage(hist1WindowDetectionsDf.loc[i,:],hist1WindowDetectionsDf.loc[x,:])

    #create a heat map of the linkage table
    plt.figure()
    sns.heatmap(linkageTable,cmap='Blues')
    plt.title('Normalized Linkage Table')
    plt.savefig('charts/co-segregation-1/linkage-table.png')
    outputFile.write('![Linkage Table heat map](../charts/co-segregation-1/linkage-table.png)\n\n')

    #write an explanation in the report
    outputFile.write('The detection frequency f(A) of a window A is the number of NPs which detect A, divided by the total number of NPs\n\n')
    outputFile.write('The co-segregation f(A,B) of two windows A and B is the number of NPs which detect both windows, divided by the total number of NPs\n\n')
    outputFile.write('The linkage disequilibrium D is equal to f(A,B)-f(A)f(B)\n\n')
    outputFile.write('The theoretical maximum of D, D-max, can be found as:\n\n')
    outputFile.write('\t min(f(A)f(B), (1-f(A))(1-f(B))) when D < 0\n\n')
    outputFile.write('\t min(f(B)(1-f(A)), f(A)(1-f(B))) when D > 0\n\n')
    outputFile.write('\t 1 when D = 0\n\n')
    outputFile.write('The normalized linkage disequilibrium, D-norm, is equal to D/D-max\n\n')
    outputFile.write('D-norm functions such that, when A and B have the maximum amount of NPs in common, it is equal to 1, ')
    outputFile.write('and when they have no NPs in common, it is -1.\n\n')

    #create a clarified heat map of the linkage table
    clarLinkageTable = linkageTable 
    for i in indices:
        for x in indices:
            val = clarLinkageTable.loc[i,x]
            if val < 0 and val != -1:
                val = -0.5
            elif val > 0 and val != 1:
                val = 0.5
            clarLinkageTable.loc[i,x] = val
    plt.figure()
    sns.heatmap(clarLinkageTable,cmap='Blues')
    plt.title('Clarified Linkage Table')
    plt.savefig('charts/co-segregation-1/clarified-linkage-table.png')
    outputFile.write('![Linkage Table heat map](../charts/co-segregation-1/clarified-linkage-table.png)\n\n') 
    outputFile.write('Linkage table with all non-integer values rounded to -0.5 and 0.5, to highlight outlying values of -1, 0, and 1\n\n')

    return 

#number of NPs in which a window is detected, divided by total numbers of NPs
def detectionFrequency(window):
    return sum(window) / len(window)

#number of NPs with both windows A and B, divided by total number of NPs
def coSegregation(windowA, windowB):
    return sum(windowA & windowB) / len(windowA)

#coSegregation(A,B) - detectionFrequency(A)*detectionFrequency(B)
def linkage(windowA, windowB):
    return coSegregation(windowA,windowB) - detectionFrequency(windowA) * detectionFrequency(windowB)

#divide linkage by its theoretical maximum
def normalizedLinkage(windowA, windowB):
    D = linkage(windowA,windowB)
    fA = detectionFrequency(windowA)
    fB = detectionFrequency(windowB)
    fAB = coSegregation(windowA,windowB)
    if D < 0:
        return D / min(fA * fB, (1-fA) * (1-fB))
    elif D > 0:
        return D / min(fB * (1-fA), fA * (1-fB))
    return 0

# D-norm will be 1 when two windows have the maximum amount of shared NPs, and -1 when they have the minimum

# Notes on normalized linkage being divided by zero:
#   - As D approaches 0, D-norm approaches 0 (if D=0, D-norm should be 0)
#   - As D-max approaches 0, D-norm approaches 1 or -1 (D-max will be 0 if a window is present in all or none of the NPs. In this
#   case, D-norm should be 1 or -1)


if __name__ == "__main__":
    main()
