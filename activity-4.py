import numpy
import math
import pandas as pd
import matplotlib.pyplot as plt
import hist1_analysis as h1_mod


def main():
    #input data file
    dataFile = 'GSE64881_segmentation_at_30000bp.passqc.multibam.txt'
    df = pd.read_csv(dataFile,sep='\t')

    #open output file
    outputFile = open("reports/activity-4-report.md", 'w')
    outputFile.write('# Activity 4 Report\n')
    outputFile.write('The Jaccard Index of two objects, A and B, with n binary attributes, determines the similarity of the objects\n\n')
    outputFile.write('The index J(A, B) can be found with the equation `|A INTERSECT B| / |A UNION B|`\n\n')

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
    print(npJaccards)

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
