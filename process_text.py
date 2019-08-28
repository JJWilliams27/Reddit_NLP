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
from operator import itemgetter
import ast
import collections
from gensim import corpora, models, similarities
import gensim
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
import logging
import pickle
# Plotting tools
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)

def list_from_string(string):
	out_list = ast.literal_eval(string)
	return out_list

def timestamp_from_string(string):
	date = datetime.datetime.strptime(string,'%Y-%m-%d %H:%M:%S').date()
	return date

def main():
	print('Processing Data')
	### Create Model ###
	full_df=pd.read_csv('Pre_Processed_DF.csv',index_col=0,parse_dates=True)

	# Extract lists from strings following df read-in
	full_df['timestamp'] = full_df['timestamp'].apply(lambda x: timestamp_from_string(x))
	full_df['Comment_Nonstop'] = full_df['Comment_Nonstop'].apply(lambda x: list_from_string(x))
	full_df['Tokenised_Comment'] = full_df['Tokenised_Comment'].apply(lambda x: list_from_string(x))
	full_df['Comment_Stemmed'] = full_df['Comment_Stemmed'].apply(lambda x: list_from_string(x))
	full_df['Comment_Lemmatized'] = full_df['Comment_Lemmatized'].apply(lambda x: list_from_string(x))

	with open("reddit_corp.cor", "rb") as cp:
		corp = pickle.load(cp) 
	with open("reddit_texts.sent", "rb") as tt:
		texts = pickle.load(tt)
    
	dictionary = gensim.corpora.dictionary.Dictionary.load("reddit_dictionary")

	min_topics = 2
	max_topics = 32
	topics_step = 2

	def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
		count = min_topics
		coherence_values = []
		perplexity_values = []
		model_list = []
		for num_topics in range(start, limit, step):
			print('Computing Model with %s' %(str(count) + ' topics'))
			model = gensim.models.ldamodel.LdaModel(corpus=corp, num_topics=num_topics, id2word=dictionary)
			model_list.append(model)
			perplexity_values.append(model.log_perplexity(corp))
			coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
			coherence_values.append(coherencemodel.get_coherence())
			count= count+topics_step

		return model_list, perplexity_values, coherence_values
	
	# Can take a long time to run.
	if compute_coherence == 1:


		model_list, perplexity_values, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corp, texts=texts, start=min_topics, limit=max_topics, step=topics_step)

		# Create Dataframe of outputs
		df_models = pd.DataFrame([x for x in perplexity_values])

		df_models['Coherence'] = coherence_values
		df_models['Num_Topics'] = range(min_topics,max_topics,topics_step) # Follows input into compute_coherence_values, start, limit, step
		df_models.columns = ["Perplexity", "Coherence", "Num_Topics"]
		df_models.to_csv('LDA_Model_Performance.csv')

		# Plot Coherence and Perplexity vs Num Topics on same graph (2 y axes)
		fig, ax1 = plt.subplots()

		y1 = np.array(df_models['Perplexity'])
		y2 = np.array(df_models['Coherence'])
		x = np.array(df_models['Num_Topics'])

		ax1.set_xlabel('Number of Topics')
		ax1.set_ylabel('Perplexity', color='b')
		ax1.plot(x, y1, 'b-')

		ax2 = ax1.twinx()
		ax2.set_ylabel('Coherence', color='r')
		ax2.plot(x, y2, 'r-')	

		ax2.spines["right"].set_edgecolor('red')
		ax2.tick_params(axis='y',colors='red')

		fig.tight_layout()
		#plt.show()

		# Select the model and print the topics
		for ii in list(range(0,len(model_list))):
			print('Processing Model with %s' %(str(df_models['Num_Topics'][ii])) + ' topics')
			optimal_model = model_list[ii]
			model_topics = optimal_model.show_topics(num_topics = df_models['Num_Topics'][ii], formatted=True, num_words = 20)
			#print(optimal_model.print_topics(num_words=20))

			# Save Model
			with open('LDA_model_topics'+ '_%s' %(str(df_models['Num_Topics'][ii])) + '.csv', 'w') as myfile:
				wr = csv.writer(myfile, quoting = csv.QUOTE_ALL)
				wr.writerow(model_topics)

			### Now Calculate Dominant Topic for Each Post/Comment ###
			def findTopics(query, lda):
				vec_bow = dictionary.doc2bow(query)
				vec_lsi = lda.get_document_topics(vec_bow)
				vec_lsi = [list(x) for x in vec_lsi]
				return(vec_lsi)

			topics_freqdist = [] # List of lists - each entry is the strength of each topic for a post/comment

			# Loop through each post/comment and work out how strong it relates to each topic, and append to a list
			for item in texts:
				try:
					topic = findTopics(item, optimal_model)
					topic.sort(key=itemgetter(1), reverse=True)
					topics_freqdist.append(topic)
				except Exception as e:
					print(e)

			model_topics_clean = []
			# Get words for each topic and clean up
			for topic in model_topics:
				topic = re.sub(r'[^a-zA-Z#]', " ", topic[1])
				topic = ' '.join(topic.split())
				model_topics_clean.append(topic)

			main_topic = []
			main_topic_strength = []
			# Get main topic for each post/comment, as well as the strength
			for topic in topics_freqdist:
				topic_no = topic[0][0]
				topic_strength = topic[0][1]
				main_topic.append(topic_no)
				main_topic_strength.append(topic_strength)
			df_temp = full_df	
			df_temp['Main_Topic'] = main_topic # Main topic for each post or comment
			df_temp['Main_Topic_Strength'] = main_topic_strength # Strength of main topic for each post or comment

			topic_words = []
	
			# Add words for each topic to dataframe
			for topic in main_topic:
				try:
					words = model_topics_clean[topic]
					topic_words.append(words)
				except:
					print(topic)
					pass

			df_temp['Topic_Words']  = topic_words

			df_temp.to_csv('topic_timeseries' + '_%s' %(str(df_models['Num_Topics'][ii])) + '.csv')

			### Find Most Representative Post/Comment for each Topic ###
			new_df = pd.DataFrame()
			new_df_small = pd.DataFrame() # Make a second dataframe with just one entry per topic, for use with output dataframe

			df_temp_grouped = df_temp.groupby('Main_Topic') # Group dataframe by topic
			
			av_sent = []
			av_score = []
			av_len = []
			for i, grp in df_temp_grouped:
				new_df = pd.concat([new_df, grp.sort_values(['Main_Topic_Strength'], ascending=[0]).head(num_topics_to_analyse)], axis = 0)
				new_df_small = pd.concat([new_df_small, grp.sort_values(['Main_Topic_Strength'], ascending=[0]).head(1)], axis = 0)
				av_sent.append(np.nanmean(grp['Sentiment'])) # Average Sentiment for each Group
				av_len.append(np.nanmean(grp['Token_Number'])) # Average No. Tokens for each Group
				scores = []
				grp.reset_index(drop=True, inplace=True)
				for ni in list(range(0,len(grp))):
					scores.append(int(grp['Score'][ni]))
				av_score.append(np.nanmean(scores)) # Average Score for each Group

			# Reset Index  
			new_df.reset_index(drop=True, inplace=True)
			new_df.to_csv('topic_naming' + '_%s' %(str(df_models['Num_Topics'][ii]))+ '.csv')
			new_df_small.reset_index(drop=True, inplace=True)
			new_df_small['Mean_Sentiment'] = av_sent
			new_df_small['Mean_Score'] = av_score
			new_df_small['Mean_Tokens'] = av_len
			### Topic Distribution Across Posts/Comments ###
			# Number of Documents for Each Topic
			topic_counts = df_temp['Main_Topic'].value_counts()

			# Percentage of Documents for Each Topic
			topic_contribution = round(topic_counts/topic_counts.sum(), 4)

			
			# Create New DataFrame
			# Concatenate Column wise
			df_dominant_topics = pd.DataFrame()
			df_dominant_topics['Main_Topic'] = new_df_small['Main_Topic']
			df_dominant_topics['Topic_Words'] = new_df_small['Topic_Words']
			df_dominant_topics = pd.concat([df_dominant_topics,topic_counts], axis=1)
			df_dominant_topics = pd.concat([df_dominant_topics,topic_contribution], axis=1)
			df_dominant_topics['Mean_Sentiment'] = new_df_small['Mean_Sentiment']
			df_dominant_topics['Mean_Score'] = new_df_small['Mean_Score']
			df_dominant_topics['Mean_Tokens'] = new_df_small['Mean_Tokens']

			df_dominant_topics.columns = ['Main_Topic','Topic_Words','Topic_Count','Topic_Percentage','Mean_Sentiment','Mean_Score','Mean_Tokens']
			df_dominant_topics.to_csv('topic_analyses' + '_%s' %(str(df_models['Num_Topics'][ii])) + '.csv')

			# Show
			df_dominant_topics

			# Save final DF
			df_temp.to_csv('final_dataframe' + '_%s' %(str(df_models['Num_Topics'][ii])) + '.csv')

			#df_temp = None
			#df_temp_grouped = None
			#new_df = None
			#new_df_small = None
			#df_dominant_topics = None

	else:
		optimal_model = gensim.models.ldamodel.LdaModel(corpus=corp,
                                           		id2word=dictionary,
                                           		num_topics=number_of_topics, 
                                           		random_state=100,
                                           		update_every=1,
                                           		chunksize=100,
                                           		passes=10,
                                           		alpha='auto',
                                           		per_word_topics=True)

		model_topics = optimal_model.show_topics(formatted=True, num_words = 20)

		# Save Model
		with open('LDA_model_topics.csv', 'w') as myfile:
			wr = csv.writer(myfile, quoting = csv.QUOTE_ALL)
			wr.writerow(model_topics)

		### Now Calculate Dominant Topic for Each Post/Comment ###
		def findTopics(query, lda):
			vec_bow = dictionary.doc2bow(query)
			vec_lsi = lda.get_document_topics(vec_bow)
			vec_lsi = [list(x) for x in vec_lsi]
			return(vec_lsi)

		topics_freqdist = [] # List of lists - each entry is the strength of each topic for a post/comment

		# Loop through each post/comment and work out how strong it relates to each topic, and append to a list
		for item in texts:
			try:
				topic = findTopics(item, optimal_model)
				topic.sort(key=itemgetter(1), reverse=True)
				topics_freqdist.append(topic)
			except Exception as e:
				print(e)

		model_topics_clean = []
		# Get words for each topic and clean up
		for topic in model_topics:
			topic = re.sub(r'[^a-zA-Z#]', " ", topic[1])
			topic = ' '.join(topic.split())
			model_topics_clean.append(topic)

		main_topic = []
		main_topic_strength = []
		# Get main topic for each post/comment, as well as the strength
		for topic in topics_freqdist:
			topic_no = topic[0][0]
			topic_strength = topic[0][1]
			main_topic.append(topic_no)
			main_topic_strength.append(topic_strength)

		full_df['Main_Topic'] = main_topic # Main topic for each post or comment
		full_df['Main_Topic_Strength'] = main_topic_strength # Strength of main topic for each post or comment

		topic_words = []

		# Add words for each topic to dataframe
		for topic in main_topic:
			try:
				words = model_topics_clean[topic]
				topic_words.append(words)
			except:
				print(topic)
				pass

		full_df['Topic_Words']  = topic_words

		### Find Most Representative Post/Comment for each Topic ###
		new_df = pd.DataFrame()
		new_df_small = pd.DataFrame() # Make a second dataframe with just one entry per topic, for use with output dataframe

		full_df_grouped = full_df.groupby('Main_Topic') # Group dataframe by topic

		for i, grp in full_df_grouped:
			new_df = pd.concat([new_df, grp.sort_values(['Main_Topic_Strength'], ascending=[0]).head(num_topics_to_analyse)], axis = 0)
			new_df_small = pd.concat([new_df_small, grp.sort_values(['Main_Topic_Strength'], ascending=[0]).head(1)], axis = 0)
    	# Reset Index    
		new_df.reset_index(drop=True, inplace=True)
		new_df.to_csv('topic_naming.csv')
		new_df_small.reset_index(drop=True, inplace=True)
		### Topic Distribution Across Posts/Comments ###
		# Number of Documents for Each Topic
		topic_counts = full_df['Main_Topic'].value_counts()

		# Percentage of Documents for Each Topic
		topic_contribution = round(topic_counts/topic_counts.sum(), 4)

		
		# Create New DataFrame


		# Concatenate Column wise
		df_dominant_topics = pd.DataFrame()
		df_dominant_topics['Main_Topic'] = new_df_small['Main_Topic']
		df_dominant_topics['Topic_Words'] = new_df_small['Topic_Words']
		df_dominant_topics = pd.concat([df_dominant_topics,topic_counts], axis=1)
		df_dominant_topics = pd.concat([df_dominant_topics,topic_contribution], axis=1)

		df_dominant_topics.columns = ['Main_Topic','Topic_Words','Topic_Count','Topic_Percentage']
		df_dominant_topics.to_csv('topic_analyses.csv')

		# Show
		df_dominant_topics

		# Save final DF
		full_df.to_csv('final_dataframe.csv')

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



















### Choose LDA Model Params ###
compute_coherence = 1 # Run LDA Model with different no. topics and compute coherence?
number_of_topics = 2 # Number of topics for LDA model if not computing coherence
num_topics_to_analyse = 50 # Number of posts/comments to analyse for each topic

if __name__ == "__main__":
	main()