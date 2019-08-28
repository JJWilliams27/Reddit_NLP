# Plot Linked Subreddits

# Import Modules
import os
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

linked_sr = pd.read_csv('Outputs/CS_FULL/LinkedSubreddits_CS_FULL.csv')

linked_sr = linked_sr.sort_values(by=['Times_Linked'],ascending=False)

# Remove link to self?
linked_sr = linked_sr[linked_sr.Subreddits != '/r/climateskeptics']
# Remove links to bots
linked_sr = linked_sr[linked_sr.Subreddits != '/r/autotldr']
linked_sr = linked_sr[linked_sr.Subreddits != '/r/autowikibot']
linked_sr = linked_sr[linked_sr.Subreddits != '/r/autowikibotsubjectglitched']
linked_sr = linked_sr[linked_sr.Subreddits != '/r/sneakpeekbot']
linked_sr = linked_sr[linked_sr.Subreddits != '/r/wikitextbot']
linked_sr = linked_sr[linked_sr.Subreddits != '/r/remindmebot']
linked_sr = linked_sr[linked_sr.Subreddits != '/r/xkcd_transcriber']
linked_sr = linked_sr[linked_sr.Subreddits != '/r/youtubot']
linked_sr = linked_sr[linked_sr.Subreddits != '/r/redditcom']
linked_sr = linked_sr[linked_sr.Subreddits != '/r/tweetposter']
linked_sr = linked_sr[linked_sr.Subreddits != '/r/helperbot_']
linked_sr = linked_sr[linked_sr.Subreddits != '/r/totesmessenger']

top20 = linked_sr[:19]

sr = top20['Subreddits']
num_links = top20['Times_Linked']
top20.reset_index(drop=True, inplace=True)

max_links = top20['Times_Linked'][0]
spacing=200
ytickvals = np.arange(0,max_links+spacing,spacing)

fig, ax = plt.subplots()
ax.text(0.9, 0.95, 'A', transform=ax.transAxes,fontsize=18, fontweight='bold', va='top')
plt.bar(sr,num_links,edgecolor='black',color='papayawhip')
plt.xticks(rotation='vertical',fontsize=12)
plt.ylabel('Frequency',fontsize=18)
plt.yticks(ytickvals)
plt.ylim(0,(max_links+(spacing/2)))
fig.tight_layout()

# Climate
linked_sr2 = pd.read_csv('Outputs/CLIM_FULL/LinkedSubreddits_CLIM.csv')

linked_sr2 = linked_sr2.sort_values(by=['Times_Linked'],ascending=False)

# Remove link to self?
linked_sr2 = linked_sr2[linked_sr2.Subreddits != '/r/climate']
# Remove links to bots
linked_sr2 = linked_sr2[linked_sr2.Subreddits != '/r/autotldr']
linked_sr2 = linked_sr2[linked_sr2.Subreddits != '/r/autowikibot']
linked_sr2 = linked_sr2[linked_sr2.Subreddits != '/r/autowikibotsubjectglitched']
linked_sr2 = linked_sr2[linked_sr2.Subreddits != '/r/sneakpeekbot']
linked_sr2 = linked_sr2[linked_sr2.Subreddits != '/r/wikitextbot']
linked_sr2 = linked_sr2[linked_sr2.Subreddits != '/r/remindmebot']
linked_sr2 = linked_sr2[linked_sr2.Subreddits != '/r/xkcd_transcriber']
linked_sr2 = linked_sr2[linked_sr2.Subreddits != '/r/youtubot']
linked_sr2 = linked_sr2[linked_sr2.Subreddits != '/r/redditcom']
linked_sr2 = linked_sr2[linked_sr2.Subreddits != '/r/tweetposter']
linked_sr2 = linked_sr2[linked_sr2.Subreddits != '/r/helperbot_']
linked_sr2 = linked_sr2[linked_sr2.Subreddits != '/r/totesmessenger']

top20_2 = linked_sr2[:19]

sr2 = top20_2['Subreddits']
num_links2 = top20_2['Times_Linked']
top20_2.reset_index(drop=True, inplace=True)

max_links = top20_2['Times_Linked'][0]
spacing=200
ytickvals = np.arange(0,max_links+spacing,spacing)

fig , ax2 = plt.subplots()
ax2.text(0.9, 0.95, 'B', transform=ax2.transAxes,fontsize=18, fontweight='bold', va='top')
plt.bar(sr2,num_links2,edgecolor='black',color='#afeeee')
plt.xticks(rotation='vertical',fontsize=12)
plt.ylabel('Frequency',fontsize=18)
plt.yticks(ytickvals)
plt.ylim(0,(max_links+(spacing/2)))
fig.tight_layout()
plt.show()