import numpy
import math
import pandas as pd
import matplotlib.pyplot as plt

#input data file
dataFile = 'GSE64881_segmentation_at_30000bp.passqc.multibam.txt'
df = pd.read_csv(dataFile,sep='\t').iloc[:,3:]

#Part 1: Estimate the radial position of each NP (apical=1 - equitorial=5)
column_sums = df.sum(axis=0)

plt.figure()
plt.scatter(column_sums.index,column_sums)
plt.savefig('column_sums.png')


column_sums = column_sums.sort_values()
np_radial_positions = pd.Series()
for index,key in enumerate(column_sums.keys()):
    np_radial_positions[key] = math.floor(index / (len(column_sums)/5)) + 1
print(np_radial_positions)


#Part 2: Estimate the compaction of each window (most condensed=1 - least condensed=10)
row_sums = df.sum(axis=1)

row_sums = row_sums.sort_values()
window_compactions = pd.Series()
for index,key in enumerate(row_sums.keys()):
    window_compactions[key] = math.floor(index / (len(row_sums)/10)) + 1
