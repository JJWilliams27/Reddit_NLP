# Plot Sentiment Comparison Figure

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

path = os.getcwd()
cs_path = path+'/Outputs/CS_FULL/Pre_Processed_DF_CS.csv'
clim_path = path+'/Outputs/CLIM_FULL/Pre_Processed_DF_CLIM.csv'

cs = pd.read_csv(cs_path)
clim = pd.read_csv(clim_path)

# Sentiment
counts,bins = np.histogram(cs['Sentiment'],bins=np.arange(-1,1.2,0.2))
total_cs = np.sum(counts)
counts2,bins2 = np.histogram(clim['Sentiment'],bins=np.arange(-1,1.2,0.2))
total_clim = np.sum(counts2)
perc_counts = (counts/total_cs)*100
perc_counts2 = (counts2/total_clim)*100
width = (bins[1] - bins[0])/3

fig, ax = plt.subplots()
bins_ax=np.arange(-1,1.2,0.2)+width
plt.bar(bins_ax[:-1],perc_counts,width,color='papayawhip',label='/r/climateskeptics')
plt.bar(bins_ax[:-1]+width,perc_counts2,width,color='#afeeee',label='/r/climate')
ax.set_xticks(bins)
plt.ylabel('Frequency (%)',fontsize=14)
plt.xlabel('Polarity',fontsize=14)
plt.legend()
plt.show()

# Calculate Percentage Positive and Negative
pos_counts = counts[5:9]
sum_pos = np.sum(pos_counts)
sum_all = np.sum(counts)
perc_cs = str((sum_pos/sum_all)*100)
print('Percentage Positive Sentiment of Skeptics (> 0.0): ' + perc_cs)

pos_counts2 = counts2[5:9]
sum_pos2 = np.sum(pos_counts2)
sum_all2 = np.sum(counts2)
perc_clim = str((sum_pos2/sum_all2)*100)
print('Percentage Positive Sentiment of Pro-Climate (> 0.0): ' + perc_clim)