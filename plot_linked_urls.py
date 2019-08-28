# Clean and Plot Linked URLS

# Import Modules
import os
import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

linked_url = pd.read_csv('Outputs/CS_FULL/LinkedURLs_CS.csv')

linked_url = linked_url.sort_values(by=['Times_Linked'],ascending=False)
linked_url.reset_index(drop=True, inplace=True)

top20 = linked_url[:19]

url = top20['URL']
num_links = top20['Times_Linked']

max_links = top20['Times_Linked'][0]
spacing=1000
ytickvals = np.arange(0,max_links+spacing,spacing)

fig, ax = plt.subplots()
ax.text(0.9, 0.95, 'A', transform=ax.transAxes,fontsize=18, fontweight='bold', va='top')
plt.bar(url,num_links,edgecolor='black',color='papayawhip')
plt.xticks(rotation='vertical',fontsize=12)
plt.ylabel('Frequency',fontsize=18)
plt.yticks(ytickvals)
plt.ylim(0,(max_links+(spacing/2)))
fig.tight_layout()

# Now for /r/climate
linked_url2 = pd.read_csv('Outputs/CLIM_FULL/LinkedURLs_CLIM.csv')

linked_url2 = linked_url2.sort_values(by=['Times_Linked'],ascending=False)
linked_url2.reset_index(drop=True, inplace=True)

top20_2 = linked_url2[:19]

url2 = top20_2['URL']
num_links2 = top20_2['Times_Linked']

max_links = top20_2['Times_Linked'][0]
spacing=1000
ytickvals = np.arange(0,max_links+spacing,spacing)

fig , ax2 = plt.subplots()
ax2.text(0.9, 0.95, 'B', transform=ax2.transAxes,fontsize=18, fontweight='bold', va='top')
plt.bar(url2,num_links2,edgecolor='black',color='#afeeee')
plt.xticks(rotation='vertical',fontsize=12)
plt.ylabel('Frequency',fontsize=18)
plt.yticks(ytickvals)
plt.ylim(0,(max_links+(spacing/2)))
fig.tight_layout()
plt.show()