# Extract Lemma, Tokens, Term Occurance (i.e. to avoid having to run the whole processing again)

import collections
import pandas as pd
import os
import matplotlib.pyplot as plt
import ast
import datetime 

# Functions
def list_from_string(string):
	out_list = ast.literal_eval(string)
	return out_list

def timestamp_from_string(string):
	date = datetime.datetime.strptime(string,'%Y-%m-%d %H:%M:%S').date()
	return date

# Read Pre Processed DF	

path = os.getcwd()
fullpath = path + '/Outputs/CS_FULL/Pre_Processed_DF_CS.csv'

full_df = pd.read_csv(fullpath)
full_df['timestamp'] = full_df['timestamp'].apply(lambda x: timestamp_from_string(x))
full_df['Comment_Nonstop'] = full_df['Comment_Nonstop'].apply(lambda x: list_from_string(x))
full_df['Tokenised_Comment'] = full_df['Tokenised_Comment'].apply(lambda x: list_from_string(x))
full_df['Comment_Stemmed'] = full_df['Comment_Stemmed'].apply(lambda x: list_from_string(x))
full_df['Comment_Lemmatized'] = full_df['Comment_Lemmatized'].apply(lambda x: list_from_string(x))

### Frequency of Terms in Corpus ###
all_words = full_df['Comment_Nonstop'].values.tolist()
all_words = [y for x in all_words for y in x]
cchange_count = 0
gw_count=0
print("Counting Term Occurance")
for i in list(range(0,len(all_words))):
	if all_words[i] == "climate":
		if all_words[i+1] == "change":
			cchange_count = cchange_count+1
	if all_words[i] == "global":
		if all_words[i+1] == "warming":
			gw_count = gw_count+1

print("Climate Change Count: " + str(cchange_count))
print("Global Warming Count: " + str(gw_count))


### PLOTS ###
# Most Common Words
# Before Stopwords Removed
print('Plotting')
all_tc = full_df['Tokenised_Comment'].values.tolist()
flattened_list = [y for x in all_tc for y in x]
counter=collections.Counter(flattened_list)
top20 = counter.most_common(20)
keys = []
values = []
for i in top20:
	keys.append(i[0])
	values.append(i[1])

token_df = pd.DataFrame()
token_df['Tokens'] = keys
token_df['Values'] = values
token_df.to_csv('Outputs/Top20_Tokens.csv')

fig=plt.figure()
plt.bar(keys,values,edgecolor='black',color='lightgreen')
plt.xticks(rotation='vertical',fontsize=12)
plt.ylabel('Frequency',fontsize=18)
plt.xlabel('Token',fontsize=18)
fig.tight_layout()
plt.show()

# After All Cleaning
all_cl = full_df['Comment_Lemmatized'].values.tolist()
flattened_list = [y for x in all_cl for y in x]
counter=collections.Counter(flattened_list)
top20 = counter.most_common(20)
keys = []
values = []
for i in top20:
	keys.append(i[0])
	values.append(i[1])

lemma_df = pd.DataFrame()
lemma_df['Lemma'] = keys
lemma_df['Values'] = values
lemma_df.to_csv('Outputs/Top20_Lemma.csv')

fig = plt.figure()	
plt.bar(keys,values,edgecolor='black',color='lightgreen')
plt.xticks(rotation='vertical',fontsize=12)
plt.ylabel('Frequency',fontsize=18)
plt.xlabel('Lemma',fontsize=18)
fig.tight_layout()

plt.show()