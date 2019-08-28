# Plot Linked Subreddit from URLs

# Import Modules
import os
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def add_r(name):
	newname = '/r/' + name
	return newname

linked_sr = pd.read_csv('Outputs/CS_FULL/LinkedURL_Subreddits_CS.csv')
linked_sr['Subreddit_Name'] = linked_sr['Subreddit'].apply(lambda x: add_r(x))

linked_sr = linked_sr.sort_values(by=['Times_Linked'],ascending=False)

# Remove link to self?
linked_sr = linked_sr[linked_sr.Subreddit != 'climateskeptics']
# Remove links to bots (i.e. autotldr, autowikibot)
linked_sr = linked_sr[linked_sr.Subreddit != 'autotldr']
linked_sr = linked_sr[linked_sr.Subreddit != 'autowikibot']
linked_sr = linked_sr[linked_sr.Subreddit != 'autowikibotsubjectglitched']
linked_sr = linked_sr[linked_sr.Subreddit != 'sneakpeekbot']
linked_sr = linked_sr[linked_sr.Subreddit != 'wikitextbot']
linked_sr = linked_sr[linked_sr.Subreddit != 'remindmebot']
linked_sr = linked_sr[linked_sr.Subreddit != 'xkcd_transcriber']
linked_sr = linked_sr[linked_sr.Subreddit != 'youtubot']
linked_sr = linked_sr[linked_sr.Subreddit != 'redditcom']
linked_sr = linked_sr[linked_sr.Subreddit != 'tweetposter']

top20 = linked_sr[:19]

sr = top20['Subreddit_Name']
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
linked_sr2 = pd.read_csv('Outputs/CLIM_FULL/LinkedURL_Subreddits_CLIM.csv')
linked_sr2['Subreddit_Name'] = linked_sr2['Subreddit'].apply(lambda x: add_r(x))

linked_sr2 = linked_sr2.sort_values(by=['Times_Linked'],ascending=False)

# Remove link to self
linked_sr2 = linked_sr2[linked_sr2.Subreddit != 'climate']
# Remove links to bots (i.e. autowikibot, autotldr)
linked_sr2 = linked_sr2[linked_sr2.Subreddit != 'autotldr']
linked_sr2 = linked_sr2[linked_sr2.Subreddit != 'autowikibot']
linked_sr2 = linked_sr2[linked_sr2.Subreddit != 'autowikibotsubjectglitched']
linked_sr2 = linked_sr2[linked_sr2.Subreddit != 'sneakpeekbot']
linked_sr2 = linked_sr2[linked_sr2.Subreddit != 'wikitextbot']
linked_sr2 = linked_sr2[linked_sr2.Subreddit != 'remindmebot']
linked_sr2 = linked_sr2[linked_sr2.Subreddit != 'xkcd_transcriber']
linked_sr2 = linked_sr2[linked_sr2.Subreddit != 'youtubot']
linked_sr2 = linked_sr2[linked_sr2.Subreddit != 'redditcom']
linked_sr2 = linked_sr2[linked_sr2.Subreddit != 'tweetposter']

top20_2 = linked_sr2[:19]

sr2 = top20_2['Subreddit_Name']
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