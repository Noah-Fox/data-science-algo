import numpy
import pandas as pd

#input data file
dataFile = 'GSE64881_segmentation_at_30000bp.passqc.multibam.txt'
myData = pd.read_csv(dataFile,sep='\t')

#count the number of windows and NPs
windowCount = len(myData['chrom'])
npCount = len(list(myData.columns))-3

print('1. Number of genomic windows:',windowCount)
print('2. Number of NPs:',npCount)

#determine counts of windows in each NP
columns = list(myData.columns)[3:]
wSum = 0
wMin = windowCount
wMax = 0
for c in columns:
    wSum += myData[c].sum()
    wMin = min(wMin,myData[c].sum())
    wMax = max(wMax,myData[c].sum())
windowAvg = wSum/len(columns)
print('3. Average number of windows in each NP:',windowAvg)
print('4. Minimum windows in an NP:',wMin)
print('   Maximum windows in an NP:',wMax)

#Determine count of NPs that each window appears in
nSum = 0
nMin = npCount 
nMax = 0 
for i in range(0,windowCount):
    rowSum = myData.iloc[i][3:].sum()
    nSum += rowSum
    nMin = min(nMin,rowSum)
    nMax = max(nMax,rowSum)
nAvg = nSum/windowCount 
print('5. Average number of NPs each window is in:',nAvg)
print('   Minimum NPs a window is in:',nMin)
print('   Maximum NPs a window is in:',nMax)
