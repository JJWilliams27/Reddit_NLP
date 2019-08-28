# Pre-Process Text Data
# Import Modules
import os
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import nltk 
import string
import re
import datetime
from tqdm import tqdm
from textblob import TextBlob
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import *
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import collections
from gensim import corpora, models, similarities
import gensim
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
import logging
import pickle
import collections
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)

analyser = SentimentIntensityAnalyzer()

def get_date(created):
    return datetime.datetime.fromtimestamp(created)

def main():
	print('Pre-Processing Reddit Data')
	print('First Pre-Process the Posts themselves')
	# Read in CSV as pandas dataframe
	data = pd.read_csv('ClimateSkepticsAllPosts.csv')
	
	### Clean Text ###

	# 1. Remove URLs from posts and make text lowercase
	def remove_url(input_txt):
		url_temp = re.findall(r"http\S*", input_txt)
		url_links.append(url_temp) # Append links to other subreddits to array for further network analysis
		input_txt = re.sub(r"http\S+", "", input_txt)
		input_txt = input_txt.lower()
		return input_txt

	url_links = [] # Set up empty array to count URLs

	data['Clean_Post'] = data['title'].apply(lambda post: remove_url(post))

	# 2. Remove Punctuation, Numbers, and Special Characters
	data['Clean_Post'] = data['Clean_Post'].str.replace("[^a-zA-Z#]", " ")

	# 3. Remove Short Words (3 Letters or Less)
	data['Clean_Post'] = data['Clean_Post'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

	# 4. Tokenisation
	data['Tokenised_Post'] = data['Clean_Post'].apply(lambda x: x.split())

	# 5. Remove Stopwords
	stopword = nltk.corpus.stopwords.words('english')
	def remove_stopwords(text):
	    text = [word for word in text if word not in stopword]
	    return text
	    
	data['Post_Nonstop'] = data['Tokenised_Post'].apply(lambda x: remove_stopwords(x))

	print('Saving to CSV')
	data.to_csv('Reddit_Post_DF_PP.csv')

	print('Read and Pre-Process Top-Level Comments and Subcomments')
	subreddit_links = []
	# Define Functions outside of loop
	def remove_deleted_comments(input_txt):
		input_txt = re.sub(r"deleted", "", input_txt)
		return input_txt

	def remove_user_subreddit(input_txt):
		input_txt = re.sub(r"/u/[\w]*", '', input_txt)
		sr_temp = re.findall(r"/r/[\w]*", input_txt)
		subreddit_links.append(sr_temp) # Append links to other subreddits to array for further network analysis
		input_txt = re.sub(r"/r/[\w]*", '', input_txt)
		return input_txt

	# Read in CSV as pandas dataframe
	toplevelcomments = pd.DataFrame(columns=['comment','timestamp','Score'])
	for i in list(range(0,len(data['id']))):
		if os.path.isfile('Comments/%s.csv' %(data['id'][i])):
			try:
				data2 = pd.read_csv('Comments/' + '%s.csv' %(data['id'][i]))
				row = next(data2.iterrows())
				tempdf = pd.DataFrame(row[1])
				tempdf['timestamp'] = pd.to_numeric(data2.iloc[2])
				tempdf['Score'] = data2.iloc[1]
				tempdf.columns = ['comment','timestamp','Score']
				toplevelcomments = pd.concat([toplevelcomments,tempdf])

				# Subcomments
				subcom_list = []
				scores = []
				subcom_ts = []
				for j in list(range(0,data2.shape[1])):
					try:
						subc = eval(data2.loc[3,:][j]) # Read text representation of dictionary as dictionary
					except:
						print(j)
						print(data2.loc[3,:][j])
					for key in subc:
						try:
							val = subc.get(key)
							score = val[0]
							comment = val[1]
							timestamp = int(val[2])
							blob = TextBlob(comment)
							sentences = blob.sentences
							temp = []
							for sentence in sentences:
								string = str(sentence)
								if string.startswith(">"): # Remove quotes of previous comments and other sources
									pass
								else:
									temp.append(str(sentence))

							new_comment = ' '.join(temp)
							subcom_list.append(new_comment)
							scores.append(score)
							subcom_ts.append(timestamp)
						except:
							pass
				if len(subcom_list)>0:
					try:
						tempsc = pd.DataFrame(subcom_list)
						tempsc.columns = ['subcomment']
						tempsc['timestamp'] = subcom_ts
						tempsc['Score'] = scores
						tempsc = tempsc[tempsc['subcomment'].map(lambda d: len(d)) > 0]
						subcomments = pd.concat([subcomments,tempsc])
					except NameError:
						subcomments = pd.DataFrame(subcom_list)
						subcomments.columns = ['subcomment']
						subcomments['timestamp'] = subcom_ts
						subcomments['Score'] = scores
						subcomments = subcomments[subcomments['subcomment'].map(lambda d: len(d)) > 0]
			except:
				pass # Pass if no comments for post	

	# Convert timestamps to Dates for comments and subcomments (posts done so in the get data phase)			
	_timestamp = toplevelcomments["timestamp"].apply(get_date)
	toplevelcomments = toplevelcomments.assign(timestamp = _timestamp)
	_timestamp = subcomments["timestamp"].apply(get_date)
	subcomments = subcomments.assign(timestamp = _timestamp)

	print('Top Level Comments: ' + str(len(toplevelcomments)))
	print('Subcomments: ' + str(len(subcomments)))
	print('Total Comments: ' + str(len(toplevelcomments) + len(subcomments)))
	print('Pre-Processing Comments')

	# 0. Remove NaNs
	toplevelcomments.dropna(axis = 0, subset=['comment'], inplace=True)
	
	# 1. Remove URLs from comments and make text lowercase. Also remove deleted comments, usernames and subreddit names.   
	toplevelcomments['Clean_Comment'] = toplevelcomments['comment'].apply(lambda x: remove_url(x))
	toplevelcomments['Clean_Comment'] = toplevelcomments['Clean_Comment'].apply(lambda x: remove_deleted_comments(x))

	toplevelcomments['Clean_Comment'] = toplevelcomments['Clean_Comment'].apply(lambda x: remove_user_subreddit(x))
	#toplevelcomments['Clean_Comment'] = toplevelcomments['Clean_Comment'].apply(lambda x: remove_climate_words(x))

	# 2. Remove Punctuation, Numbers, and Special Characters
	toplevelcomments['Clean_Comment'] = toplevelcomments['Clean_Comment'].str.replace("[^a-zA-Z#]", " ")

	# 3. Remove Short Words (3 Letters or Less)
	toplevelcomments['Clean_Comment'] = toplevelcomments['Clean_Comment'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

	# 4. Tokenisation
	toplevelcomments['Tokenised_Comment'] = toplevelcomments['Clean_Comment'].apply(lambda x: x.split())

	# 5. Remove Stopwords		    
	toplevelcomments['Comment_Nonstop'] = toplevelcomments['Tokenised_Comment'].apply(lambda x: remove_stopwords(x))

	# 6. Remove Blank Rows from Dataframe
	toplevelcomments = toplevelcomments[toplevelcomments['Comment_Nonstop'].map(lambda d: len(d)) > 0] # Only keep rows where tokenised comments has at least 1 element

	path=os.getcwd()
	dirname = path + '/Comments_PP'
	if not os.path.exists(dirname):
		os.mkdir(dirname)
	toplevelcomments.to_csv('Comments_PP/' + 'All_TLC_PP.csv')

	# Clean Subcomments
	# 0. Remove NaNs
	subcomments.dropna(axis = 0, subset=['subcomment'], inplace=True)

	# 1. Remove URLs from comments and make text lowercase. Also remove deleted comments, usernames and subreddit names.    
	subcomments['Clean_Comment'] = subcomments['subcomment'].apply(lambda x: remove_url(x))
	subcomments['Clean_Comment'] = subcomments['Clean_Comment'].apply(lambda x: remove_deleted_comments(x))

	subcomments['Clean_Comment'] = subcomments['Clean_Comment'].apply(lambda x: remove_user_subreddit(x))
	#subcomments['Clean_Comment'] = subcomments['Clean_Comment'].apply(lambda x: remove_climate_words(x))

	# 2. Remove Punctuation, Numbers, and Special Characters
	subcomments['Clean_Comment'] = subcomments['Clean_Comment'].str.replace("[^a-zA-Z#]", " ")

	# 3. Remove Short Words (3 Letters or Less)
	subcomments['Clean_Comment'] = subcomments['Clean_Comment'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

	# 4. Tokenisation
	subcomments['Tokenised_Comment'] = subcomments['Clean_Comment'].apply(lambda x: x.split())

	# 5. Remove Stopwords		    
	subcomments['Comment_Nonstop'] = subcomments['Tokenised_Comment'].apply(lambda x: remove_stopwords(x))

	# 6. Remove Blank Rows from Dataframe
	subcomments = subcomments[subcomments['Comment_Nonstop'].map(lambda d: len(d)) > 0] # Only keep rows where tokenised comments has at least 1 element

	path=os.getcwd()
	dirname = path + '/Comments_PP'
	if not os.path.exists(dirname):
		os.mkdir(dirname)

	subcomments.to_csv('Comments_PP/' + 'All_SC_PP.csv')

	## Combine Dataframes
	# Extract the key columns from main post dataframe
	posts = data[['title','timestamp','score','Clean_Post','Tokenised_Post','Post_Nonstop']]
	# Make sure column names are the same
	posts.columns = ['Comment','timestamp','Score','Clean_Comment','Tokenised_Comment','Comment_Nonstop']
	toplevelcomments.columns = ['Comment','timestamp','Score','Clean_Comment','Tokenised_Comment','Comment_Nonstop']
	subcomments.columns = ['Comment','timestamp','Score','Clean_Comment','Tokenised_Comment','Comment_Nonstop']

	full_df = pd.concat([posts,toplevelcomments,subcomments])
	full_df.reset_index(drop=True, inplace=True)

	print('Get Links to other Subreddits')
	linked_subreddits = list(filter(None,subreddit_links))
	def clean_linked_sr(linked_subreddits):
		return[y for x in linked_subreddits for y in x]
	linked_subreddits = clean_linked_sr(linked_subreddits)

	counter = collections.Counter(linked_subreddits)
	top20 = counter.most_common(20)
	keys = []
	values = []
	for i in top20:
		keys.append(i[0])
		values.append(i[1])
	counter=dict(counter)
	linked_subreddits = list(set(linked_subreddits))

	sr_links = []
	for sr in linked_subreddits:
		num_links = counter.get(sr)
		sr_links.append(num_links)

	linked_sr = pd.DataFrame()
	linked_sr['Subreddits'] = linked_subreddits
	linked_sr['Times_Linked'] = sr_links

	linked_sr.to_csv('LinkedSubreddits.csv')

	fig = plt.figure()
	plt.bar(keys,values,edgecolor='black',color='aquamarine')
	plt.xticks(rotation='vertical')
	plt.ylabel('Number of Links',fontsize=18)
	plt.xlabel('Subreddit',fontsize=18)
	fig.tight_layout()

	print('Get URLs')
	linked_urls = list(filter(None,url_links))

	def clean_linked_url(linked_urls):
		return[y for x in linked_urls for y in x]

	def get_subreddit_from_url(link):
		components = link.split('.')
		if 'reddit' in components:
			try:
				parts = link.split('/')
				idx = parts.index('r')
				idx = idx+1
				sr = parts[idx]
				sr = re.sub(r'\W+', '', sr) # Remove non alphanumeric characters (to avoid instances of i.e. subredditname) or subredditname*) 
				sr = sr.lower() # Make lowercase to avoid case-based duplicates (i.e. ecointernet, EcoInternet)
				return sr
			except:
				pass # Skip other uses of reddit url (i.e. reddit.com/message/xyz)

	def get_main_url(link):
		try:
			return link.split('/')[2]
		except:
			try:
				return link.split('/')[1] # Sometimes http:/www.website.com
			except:
				pass # Skip if error

	linked_urls = clean_linked_url(linked_urls)
	url_sr = []
	clean_urls = []
	for link in linked_urls:
		sr_url = get_subreddit_from_url(link)
		url_sr.append(sr_url)
		link = get_main_url(link)
		try:
			if link.startswith('www.'): # Make sure all links start with www. (to avoid miscounting in the instance of having websitename.com and www.websitename.com)
				clean_urls.append(link)
			else:
				link = 'www.' + link
				clean_urls.append(link)
		except:
			pass # Skip empty entries
	url_sr = list(filter(None, url_sr)) # Remove None from URL subreddits 
	linked_url_subreddits = list(set(url_sr))

	counter = collections.Counter(clean_urls)
	top20 = counter.most_common(20)
	keys = []
	values = []
	for i in top20:
		keys.append(i[0])
		values.append(i[1])
	counter=dict(counter)
	linked_urls = list(set(clean_urls))

	url_links = []
	for url in linked_urls:
		num_links = counter.get(url)
		url_links.append(num_links)

	fig = plt.figure()
	plt.bar(keys,values,edgecolor='black',color='aquamarine')
	plt.xticks(rotation='vertical')
	plt.ylabel('Number of Links',fontsize=18)
	plt.xlabel('URL',fontsize=18)
	fig.tight_layout()

	counter = collections.Counter(url_sr)
	counter=dict(counter)
	url_sr_links = []
	for sr in linked_url_subreddits:
		num_links = counter.get(sr)
		url_sr_links.append(num_links)

	linked_url = pd.DataFrame()
	linked_url['URL'] = linked_urls
	linked_url['Times_Linked'] = url_links
	linked_url.drop_duplicates(subset ="URL", keep = 'first', inplace = True)

	linked_url.to_csv('LinkedURLs.csv')

	linked_url_sr = pd.DataFrame()
	linked_url_sr['Subreddit'] = linked_url_subreddits
	linked_url_sr['Times_Linked'] = url_sr_links
	linked_url_sr.drop_duplicates(subset ="Subreddit", keep = 'first', inplace = True)
	pdb.set_trace()
	linked_url_sr.to_csv('LinkedURL_Subreddits.csv')
	
	fig = plt.figure()
	plt.bar(keys,values,edgecolor='black',color='crimson')
	plt.xticks(rotation='vertical')
	plt.ylabel('Number of Links to Subreddits in URLs',fontsize=18)
	fig.tight_layout()

	## Create Bigrams ##
	if use_bigrams == 1:
		all_words = full_df['Comment_Nonstop'].values.tolist()
		phrases = Phrases(all_words, min_count=30)
		bigram=Phraser(phrases)

		def make_bigrams(texts):
			return[bigram[texts]]
		print("Creating Bigrams")
		full_df['Bigrams'] = full_df['Comment_Nonstop'].apply(lambda x: make_bigrams(x))

		def clean_bigrams(texts):
			return[y for x in texts for y in x]

		full_df['Bigrams'] = full_df['Bigrams'].apply(lambda x: clean_bigrams(x))

	## Stemming and Lemmatization ##
	stemmer = PorterStemmer()
	def stemming(text):
	    text = [stemmer.stem(word) for word in text]
	    return text

	if use_bigrams == 1:
		full_df['Comment_Stemmed'] = full_df['Bigrams'].apply(lambda x: stemming(x))
	else:
		full_df['Comment_Stemmed'] = full_df['Comment_Nonstop'].apply(lambda x: stemming(x))

	wn = nltk.WordNetLemmatizer()
	def lemmatizer(text):
		text = [wn.lemmatize(word) for word in text]
		return text

	if use_bigrams == 1:
		full_df['Comment_Lemmatized'] = full_df['Bigrams'].apply(lambda x: lemmatizer(x))
	else:
		full_df['Comment_Lemmatized'] = full_df['Comment_Nonstop'].apply(lambda x: lemmatizer(x))

	## Calculate Sentiment for Every Comment, as well as number of tokens ##
	print('Calculate Sentiment')
	sent_list = []
	simp_sent_list = []
	len_comm = []
	for comm in list(range(0,len(full_df))):
		comm_sent = analyser.polarity_scores(full_df['Clean_Comment'][comm])['compound']
		sent_list.append(comm_sent)
		if comm_sent >= 0.05:
			simp_sent_list.append(1)
		elif comm_sent <= -0.05:
			simp_sent_list.append(-1)
		else:
			simp_sent_list.append(0)
		comm_len = len(full_df['Tokenised_Comment'][comm])
		len_comm.append(comm_len)

	full_df['Sentiment'] = sent_list
	full_df['Simple_Sentiment'] = simp_sent_list
	full_df['Token_Number'] = len_comm

	## Remove Duplicates ##
	duplicates = []
	print('Removing Remaining Duplicates')
	full_df.drop_duplicates(subset ="Clean_Comment", keep = 'first', inplace = True) 
	print('Saving Pre-Processed Data')
	full_df.to_csv('Pre_Processed_DF.csv')
	print('DF Saved')

	## Create Corpus ##
	comments = list(full_df['Comment_Lemmatized'])
	all_tokens = [j for i in comments for j in i]

	# Remove words that appears twice or less (i.e. Jacobi et al 2016; Mountford 2018)
	tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1 or all_tokens.count(word) == 2)
	print('Get Texts')
	texts = [[word for word in text if word not in tokens_once] for text in comments]
	print("Total Number of Posts and Comments: "+str(len(texts)))
	print("Total Number of Words: " + (str(len(all_tokens) - len(tokens_once))))
	print('Create Dictionary')
	dictionary = corpora.Dictionary(texts)
	print('Create Corp')
	corp = [dictionary.doc2bow(text) for text in texts]

	# Human readable format of corpus (term-frequency)
	# print([[(dictionary[id], freq) for id, freq in cp] for cp in corp[:]])
	dictionary.save("reddit_dictionary")
	with open("reddit_corp.cor", "wb") as fp: 
		pickle.dump(corp, fp)
	with open("reddit_texts.sent","wb") as tt:
		pickle.dump(texts, tt)
	print('Data Pre-Processed')

### Options ###
use_bigrams = 0 # Compute bigrams? This causes a problem where it will link climate and change to form climate_change, but these words also occur separately. This results in modelled topics with the words climate_change, climate, change

if __name__ == "__main__":
	main()