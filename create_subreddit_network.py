# Create Subreddit network using networkx

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
import collections
from tqdm import tqdm
from itertools import count
import networkx as nx
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

input_subreddits = ['ClimateSkeptics','Climate']

# Set minimum number of individual users and submissions required for a subreddit to be added to the network
min_users = 20
min_submissions = 1000


# Read in csv
cs_df = pd.read_csv('climateskeptics_SubredditNetwork_FULL.csv')
co_df = pd.read_csv('climate_SubredditNetwork_FULL.csv')

cs_df = cs_df[cs_df['Distinct_Users'] > min_users]
cs_df = cs_df[cs_df['Total_Submissions'] > min_submissions]
co_df = co_df[co_df['Distinct_Users'] > min_users]
co_df = co_df[co_df['Total_Submissions'] > min_submissions]

# Create Full DataFrame with All Data

# Need to split up skeptics and activists
cs_df['CS_Submissions'] = cs_df['Total_Submissions']
cs_df['CS_Users'] = cs_df['Distinct_Users']
cs_df['CO_Submissions'] = 0
cs_df['CO_Users'] = 0

co_df['CS_Submissions'] = 0
co_df['CS_Users'] = 0
co_df['CO_Submissions'] = co_df['Total_Submissions']
co_df['CO_Users'] = co_df['Distinct_Users']

# Combine dataframes and merge, grouping by subreddits and summing the other columns
full_df = pd.concat([cs_df,co_df],ignore_index=True)
full_df = full_df.groupby(['Subreddits']).sum().reset_index()

full_df = full_df.sort_values(by=['Distinct_Users'],ascending=False)
full_df.reset_index(drop=True, inplace=True)

cs_nodes = list(cs_df['Subreddits'])
co_nodes = list(co_df['Subreddits'])


# Create Graph
G = nx.Graph()
G.add_node('ClimateSkeptics', perc_cs = 100, totalsubs=60000) # Add home node
G.add_node('Climate',perc_cs = 0, totalsubs=60000)

for n in cs_nodes:
	idx = full_df.loc[full_df['Subreddits'] == n].index[0]
	total_subs = full_df['Total_Submissions'][idx]
	perc_cs_users = (full_df['CS_Users'][idx]/full_df['Distinct_Users'][idx])*100
	G.add_node(n,perc_cs = float(perc_cs_users), totalsubs = total_subs)
	G.add_edge('ClimateSkeptics',n,weight=full_df['CS_Users'][idx])

for n in co_nodes:
	idx = full_df.loc[full_df['Subreddits'] == n].index[0]
	total_subs = full_df['Total_Submissions'][idx]
	perc_cs_users = (full_df['CS_Users'][idx]/full_df['Distinct_Users'][idx])*100
	G.add_node(n,perc_cs = float(perc_cs_users), totalsubs = total_subs)
	G.add_edge('Climate',n,weight=full_df['CO_Users'][idx])

# Create Network Plot
nodes = G.nodes()
colors = [G.node[n]['perc_cs'] for n in nodes]
sizes = [(G.node[n]['totalsubs']/50) for n in nodes]

fig = plt.figure()
pos = nx.spring_layout(G)
ec = nx.draw_networkx_edges(G, pos, alpha=0.2)
nc = nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors, node_size=sizes, with_labels=True, cmap=plt.cm.RdYlGn_r, alpha=0.95)
nx.draw_networkx_labels(G, pos=pos, font_size=10)
cbar = plt.colorbar(nc)
cbar.set_label("% Users from /r/climateskeptics", rotation = 90, fontsize=16)
plt.axis('off')
fig.tight_layout()
plt.show()

# Export to GEPHI Format
#nx.write_gexf(G, "test.gexf")