# Time series subplots

# Import Modules
import os
import pandas as pd
import numpy as np
import datetime as dt
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)

# All Posts
# Read CSV
df = pd.read_csv('ClimateSkepticsAllPosts.csv',index_col=0,parse_dates=True)

def get_yearmonth(timestamp):
	month = timestamp[5:7]
	year  = timestamp[:4]
	monthyear = str(year) + '/' + str(month)
	return monthyear

df['YearMonth'] = df['timestamp'].apply(lambda x: get_yearmonth(x)) # Bin Data in Months

df_grp = df.groupby('YearMonth')

Y_M = []
Num_Posts = []

for i, grp in df_grp:
	grplen = len(grp)
	Num_Posts.append(grplen) # Get Number of Posts per Month
	Y_M.append(i)

# New DateFrame
dateTime = pd.DataFrame()
dateTime['YearMonth'] = Y_M
dateTime['Posts'] = Num_Posts

dateTime.to_csv('ClimateSkeptics_Posts_per_Month.csv')

datetime_list = []
for i in list(range(0,len(dateTime))):
	month = int(dateTime['YearMonth'][i][5:7])
	year = int(dateTime['YearMonth'][i][0:4])

	datetime_list.append(dt.date(year,month,1))

dateTime['Date'] = datetime_list


# All Submissions
# Read CSV
path=os.getcwd()
fullpath=path+'/Outputs/CS_FULL/LDA_Dataframes/topic_timeseries_10.csv'
df = pd.read_csv(fullpath,index_col=0,parse_dates=True)

df['YearMonth'] = df['timestamp'].apply(lambda x: get_yearmonth(x)) # Bin Data in Months

df_grp = df.groupby('YearMonth')

Y_M = []
Num_Posts = []

for i, grp in df_grp:
	grplen = len(grp)
	Num_Posts.append(grplen) # Get Number of Posts per Month
	Y_M.append(i)

# New DateFrame
dateTime2 = pd.DataFrame()
dateTime2['YearMonth'] = Y_M
dateTime2['Posts'] = Num_Posts

dateTime2.to_csv('ClimateSkeptics_Submissions_per_Month.csv')

datetime_list = []
for i in list(range(0,len(dateTime2))):
	month = int(dateTime2['YearMonth'][i][5:7])
	year = int(dateTime2['YearMonth'][i][0:4])

	datetime_list.append(dt.date(year,month,1))

dateTime2['Date'] = datetime_list

# Get Subscribers
subs = pd.read_csv('climateskeptics_subscribers.csv')
subs.columns = ['timestamp','Subscribers']

datetime_list = []
for i in list(range(0,len(subs))):
	day = subs['timestamp'][i][:2]
	month = subs['timestamp'][i][3:5]
	year = subs['timestamp'][i][6:10]
	datetime_list.append(dt.date(int(year),int(month),int(day)))

subs['Date'] = datetime_list


# NOW DO SPECIFIC SEARCHES
# CLIMATEGATE

cgate_posts = pd.read_csv('CS_CGate_posts.csv')
cgate_posts = cgate_posts.drop(['title','url','comms_num'],axis=1)
cgate_coms = pd.read_csv('CS_CGate_comments.csv')

cgate_df = pd.concat([cgate_posts,cgate_coms])

cgate_df['YearMonth'] = cgate_df['timestamp'].apply(lambda x: get_yearmonth(x)) # Bin Data in Months
cgate_df.drop_duplicates(subset ="id", keep = 'first', inplace = True) # Remove duplicates based on ID

cgate_df_grp = cgate_df.groupby('YearMonth')

Y_M = []
Num_Posts = []

for i, grp in cgate_df_grp:
	grplen = len(grp)
	Num_Posts.append(grplen) # Get Number of Posts per Month
	Y_M.append(i)

# New DateFrame
CG_dateTime = pd.DataFrame()
CG_dateTime['YearMonth'] = Y_M
CG_dateTime['Posts'] = Num_Posts

datetime_list = []
for i in list(range(0,len(CG_dateTime))):
	month = int(CG_dateTime['YearMonth'][i][5:7])
	year = int(CG_dateTime['YearMonth'][i][0:4])

	datetime_list.append(dt.date(year,month,1))

CG_dateTime['Date'] = datetime_list

# IPCC AR4

ipcc_posts = pd.read_csv('CS_IPCC_posts.csv')
ipcc_posts = ipcc_posts.drop(['title','url','comms_num'],axis=1)
ipcc_coms = pd.read_csv('CS_IPCC_comments.csv')

ipcc_df = pd.concat([ipcc_posts,ipcc_coms])

ipcc_df['YearMonth'] = ipcc_df['timestamp'].apply(lambda x: get_yearmonth(x)) # Bin Data in Months
ipcc_df.drop_duplicates(subset ="id", keep = 'first', inplace = True) # Remove duplicates based on ID

ipcc_df_grp = ipcc_df.groupby('YearMonth')

Y_M = []
Num_Posts = []

for i, grp in ipcc_df_grp:
	grplen = len(grp)
	Num_Posts.append(grplen) # Get Number of Posts per Month
	Y_M.append(i)

# New DateFrame
IPCC_dateTime = pd.DataFrame()
IPCC_dateTime['YearMonth'] = Y_M
IPCC_dateTime['Posts'] = Num_Posts

datetime_list = []
for i in list(range(0,len(IPCC_dateTime))):
	month = int(IPCC_dateTime['YearMonth'][i][5:7])
	year = int(IPCC_dateTime['YearMonth'][i][0:4])

	datetime_list.append(dt.date(year,month,1))

IPCC_dateTime['Date'] = datetime_list

# Paris COP21

cop21_posts = pd.read_csv('CS_COP21_posts.csv')
cop21_posts = cop21_posts.drop(['title','url','comms_num'],axis=1)
cop21_coms = pd.read_csv('CS_COP21_comments.csv')

cop21_df = pd.concat([cop21_posts,cop21_coms])
cop21_df.drop_duplicates(subset ="id", keep = 'first', inplace = True) # Remove duplicates based on ID

cop21_df['YearMonth'] = cop21_df['timestamp'].apply(lambda x: get_yearmonth(x)) # Bin Data in Months

cop21_df_grp = cop21_df.groupby('YearMonth')

Y_M = []
Num_Posts = []

for i, grp in cop21_df_grp:
	grplen = len(grp)
	Num_Posts.append(grplen) # Get Number of Posts per Month
	Y_M.append(i)

# New DateFrame
COP21_dateTime = pd.DataFrame()
COP21_dateTime['YearMonth'] = Y_M
COP21_dateTime['Posts'] = Num_Posts

datetime_list = []
for i in list(range(0,len(COP21_dateTime))):
	month = int(COP21_dateTime['YearMonth'][i][5:7])
	year = int(COP21_dateTime['YearMonth'][i][0:4])

	datetime_list.append(dt.date(year,month,1))

COP21_dateTime['Date'] = datetime_list

# Cooling/Snow/Freeze/Cold

cold_posts = pd.read_csv('CS_cooling_posts.csv')
cold_posts = cold_posts.drop(['title','url','comms_num'],axis=1)
cold_coms = pd.read_csv('CS_cooling_comments.csv')

cold_df = pd.concat([cold_posts,cold_coms])
cold_df.drop_duplicates(subset ="id", keep = 'first', inplace = True) # Remove duplicates based on ID
cold_df['YearMonth'] = cold_df['timestamp'].apply(lambda x: get_yearmonth(x)) # Bin Data in Months

cold_df_grp = cold_df.groupby('YearMonth')

Y_M = []
Num_Posts = []

for i, grp in cold_df_grp:
	grplen = len(grp)
	Num_Posts.append(grplen) # Get Number of Posts per Month
	Y_M.append(i)

# New DateFrame
cold_dateTime = pd.DataFrame()
cold_dateTime['YearMonth'] = Y_M
cold_dateTime['Posts'] = Num_Posts

datetime_list = []
for i in list(range(0,len(cold_dateTime))):
	month = int(cold_dateTime['YearMonth'][i][5:7])
	year = int(cold_dateTime['YearMonth'][i][0:4])

	datetime_list.append(dt.date(year,month,1))

cold_dateTime['Date'] = datetime_list


# Plot
fig = plt.figure()
ax = plt.subplot(221)
ax.text(0.02, 0.95, 'A', transform=ax.transAxes,fontsize=16, fontweight='bold', va='top')
p1 = dateTime.plot(x='Date',y='Posts',linewidth=3,legend=False,fontsize=10,color='coral',ax=ax)
p1.set_ylabel("Number of Posts", fontsize=14)
p1.set_xlabel("Date", fontsize=14)

ax2 = plt.subplot(222)
ax2.text(0.02, 0.95, 'B', transform=ax2.transAxes,fontsize=16, fontweight='bold', va='top')
p2 = dateTime2.plot(x='Date',y='Posts',linewidth=3,legend=False,fontsize=10,color='crimson',ax=ax2)
p2.set_ylabel("Number of Submissions", fontsize=14)
p2.set_xlabel("Date", fontsize=14)

ax3 = plt.subplot(223)
ax3.text(0.02, 0.95, 'C', transform=ax3.transAxes,fontsize=16, fontweight='bold', va='top')
p3 = subs.plot(x='Date',y='Subscribers',linewidth=3,legend=False,fontsize=10,color='lightskyblue',ax=ax3)
p3.set_xlim(min(dateTime['Date']), max(dateTime['Date']))
p3.set_xlabel("Date",fontsize=14)
p3.set_ylabel("Subscribers",fontsize=14)

ax4 = plt.subplot(224)
ax4.text(0.02, 0.95, 'D', transform=ax4.transAxes,fontsize=16, fontweight='bold', va='top')
p4 = CG_dateTime.plot(x='Date',y='Posts',legend=False,fontsize=10,color='blue',ax=ax4,label='"Climategate"')
IPCC_dateTime.plot(x='Date',y='Posts',legend=False,fontsize=10,color='red',ax=ax4,label='"IPCC"')
COP21_dateTime.plot(x='Date',y='Posts',legend=False,fontsize=10,color='magenta',ax=ax4,label='"Paris"')
cold_dateTime.plot(x='Date',y='Posts',legend=False,fontsize=10,color='cyan',ax=ax4,label='"Cooling"')
ax4.set_xlabel("Date", fontsize=14)
ax4.set_ylabel("Number of Submissions", fontsize=14)
ax4.legend(loc='upper right')

#fig.tight_layout()
plt.show()