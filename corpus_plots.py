# Plot Top 20 Lemma, LDA Box Plots

# Import Modules
import os
import pandas as pd
import numpy as np
import csv
import pdb
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

path=os.getcwd()
filepath = path+'/Outputs/CS_FULL/LDA_Dataframes/final_dataframe_10.csv'

# Read full DataFrame
full_df = pd.read_csv(filepath)

# Top 20 Lemma (ClimateSkeptics)
t20l = pd.read_csv('Outputs/CS_FULL/Top20_Lemma_CS.csv')
freq = t20l['Values']
lem = t20l['Lemma']
fig, ax = plt.subplots()
ax.text(0.9, 0.95, 'A', transform=ax.transAxes,fontsize=18, fontweight='bold', va='top')	
plt.bar(lem,freq,edgecolor='black',color='papayawhip')
plt.xticks(rotation='vertical',fontsize=12)
plt.ylabel('Frequency',fontsize=18)
#plt.xlabel('Lemma',fontsize=18)
fig.tight_layout()

# Top 20 Lemma (Climate)
t20l_clim = pd.read_csv('Outputs/CLIM_FULL/Top20Lemma_CLIM.csv')
freq2 = t20l_clim['Values']
lem2 = t20l_clim['Lemma']
fig, ax = plt.subplots()	
ax.text(0.9, 0.95, 'B', transform=ax.transAxes,fontsize=18, fontweight='bold', va='top')
plt.bar(lem2,freq2,edgecolor='black',color='#afeeee')
plt.xticks(rotation='vertical',fontsize=12)
plt.ylabel('Frequency',fontsize=18)
#plt.xlabel('Lemma',fontsize=18)
fig.tight_layout()



# Topic Frequency
full_df['Main_Topic']+=1 # Add 1 to all topic numbers to avoid Topic 0
plt.figure()
full_df['Main_Topic'].plot(kind='hist',bins=np.arange(max(full_df['Main_Topic']+2))-0.5,edgecolor='k',color='papayawhip')
plt.xlabel("Topic", fontsize=16)
plt.ylabel("Frequency", fontsize=16)
plt.xticks(np.arange(1,max(full_df['Main_Topic'])+1,1)) ## MAYBE EDIT TO HAVE TOPIC NAMES HERE

# Box Plots
cs_grp = full_df.groupby('Main_Topic')
top_sent_list_cs = []
top_score_list_cs = []
top_len_list_cs = []
for i, grp in cs_grp:
	temp = grp['Sentiment']
	top_sent_list_cs.append(temp)
	temp=grp['Score']
	top_score_list_cs.append(temp)
	temp=grp['Token_Number']
	top_len_list_cs.append(temp)

# Set properties
medianprops = dict(linestyle='-',linewidth=1,color='black')
get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(n))) # Get random colours
colors = ['lightcoral','palegreen','lavender','paleturquoise','bisque','gainsboro','powderblue','lightsalmon','aquamarine','lemonchiffon']
#colors=get_colors(max(co['Main_Topic'])) # Get number of random colours depending on number of topics

# ClimateSkeptics
plt.figure()
ax = plt.subplot(3,1,1)
medianprops = dict(linestyle='-',linewidth=1,color='black')
bp = plt.boxplot(top_sent_list_cs,vert=True,patch_artist=True,medianprops=medianprops,sym='')
plt.ylabel("Polarity",fontsize=14)
plt.xticks([])
#ax.yaxis.grid(True)
# fill with colors
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

ax = plt.subplot(3,1,2)
bp2 = plt.boxplot(top_score_list_cs,vert=True,patch_artist=True,medianprops=medianprops,sym='')
plt.ylabel("Score",fontsize=14)
plt.xticks([])
ax.set_ylim(-10,15)
for patch, color in zip(bp2['boxes'], colors):
    patch.set_facecolor(color)

ax = plt.subplot(3,1,3)
bp3 = plt.boxplot(top_len_list_cs,vert=True,patch_artist=True,medianprops=medianprops,sym='')
plt.xlabel("Topic",fontsize=14)
plt.ylabel("No. Tokens",fontsize=14)
ax.set_ylim(0,130)
for patch, color in zip(bp3['boxes'], colors):
    patch.set_facecolor(color)


plt.tight_layout()
plt.show()
