'''
Extract posts from a specified subreddit, and extract all comments from each post

Author: Josh Williams
Date: 18/06/2019
Update: 18/06/2019
'''

# Import Modules
import praw
from psaw import PushshiftAPI
import pandas as pd
import datetime as dt
import os

# Options
save_posts = 1
save_comments = 1
get_top_submissions = 0
get_all_submissions = 1
get_comments_for_timeseries = 0
get_submissions_for_timeseries = 0

# All Posts
start_epoch = int(dt.datetime(2008, 1, 1).timestamp()) # Set start point for post extraction
number_of_submissions = None # Set number of posts (None = all posts)

# Create Functions
def get_date(created):
    return dt.datetime.fromtimestamp(created)

# Set up Reddit API
reddit = praw.Reddit(client_id='INSERT_CLIENT_ID_HERE', \
                     client_secret='INSERT_CLIENT_SECRET_HERE', \
                     user_agent='INSERT_USER_AGENT_HERE', \
                     username='INSERT_USERNAME_HERE', \
                     password='INSERT_PASSWORD HERE')

api = PushshiftAPI(reddit) # Use Pushshift API to get around 1000 submission limit imposed by praw

# Access Climate Skepticism Subreddit
subreddit = reddit.subreddit('ClimateSkeptics')


# Loop through top submissions and append to output dataframe
if get_top_submissions == 1:
	# Create Output Dictionary
	topics_dict = { "title":[], \
                "score":[], \
                "id":[], "url":[], \
                "comms_num": [], \
                "created": [], \
                "body":[]}
	# Access Top x posts
	print("Retrieving Submissions")
	top_subreddit = subreddit.top(limit=500)

	print("Appending Submissions to Dataframe")
	count = 0
	for submission in top_subreddit:
		print(count)
		path = os.getcwd()
		conversedict = {}
		dirname = path + '/Comments'
		if not os.path.exists(dirname):
			os.mkdir(dirname)
		outname = dirname + '/' + submission.id + '.csv'
		# Remove limit on comment extraction                
		submission.comments.replace_more(limit=None)
		topics_dict["title"].append(submission.title)
		topics_dict["score"].append(submission.score)
		topics_dict["id"].append(submission.id)
		topics_dict["url"].append(submission.url)
		topics_dict["comms_num"].append(submission.num_comments)
		topics_dict["created"].append(submission.created)
		topics_dict["body"].append(submission.selftext)
		temp_array = []
		for comment in submission.comments.list():
			temp_array.append(comment)
			if comment.id not in conversedict:
				comment.created = get_date(comment.created)
				conversedict[comment.id] = [comment.body,comment.ups,comment.created,{}] # Original = [comment.body,{}]
				if comment.parent() != submission.id:
					parent = str(comment.parent())
					conversedict[parent][3][comment.id] = [comment.ups, comment.body, comment.created]
				#conversedict[comment.id] = [comment.body,{}]
				#if comment.parent() != submission.id:
			#		parent = str(comment.parent())
		#			pdb.set_trace()
	#				conversedict[parent][1][comment.id] = [comment.ups, comment.body]

		converse_df = pd.DataFrame(conversedict)
		count = count+1
		if save_comments == 1:
			converse_df.to_csv('%s' %(outname), index=False)

	# Convert Dictionary to Pandas Dataframe
	print("Creating Dataframe")
	topics_data = pd.DataFrame(topics_dict)

	# Convert Date to Timestamp
	_timestamp = topics_data["created"].apply(get_date)
	topics_data = topics_data.assign(timestamp = _timestamp)

	# Export as CSV
	if save_posts == 1:
		print("Saving as csv")
		topics_data.to_csv('%sTop500Posts_Test.csv' %(subreddit), index=False) 



if get_all_submissions == 1:
	years=[2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]
	total_posts = []
	for year in years:
		print('Getting Submissions for %s' %(year))
		start_epoch = int(dt.datetime(year, 1, 1).timestamp()) # Set start point for post extraction
		end_epoch = int(dt.datetime(year,12,31).timestamp()) # Set end point
		# Create Output Dictionary
		topics_dict = { "title":[], \
	                "score":[], \
	                "id":[], "url":[], \
	                "comms_num": [], \
	                "created": [], \
	                "body":[]}
			# Access Top x posts
		print("Retrieving Submissions")
		all_subreddit = list(api.search_submissions(before=end_epoch,after=start_epoch,subreddit=subreddit,filter=['url','author','title','subreddit'],limit=number_of_submissions))
		total_posts.append(len(all_subreddit))
		print("Appending Submissions to Dataframe")
		count = 1
		num = len(all_subreddit)
		for submission in all_subreddit:
			print(str(count) + '/' + str(num))
			path = os.getcwd()
			dirname = path + '/Comments'
			conversedict = {}
			if not os.path.exists(dirname):
				os.mkdir(dirname)
			outname = dirname + '/' + submission.id + '.csv'
			# Remove limit on comment extraction                
			topics_dict["title"].append(submission.title)
			topics_dict["score"].append(submission.score)
			topics_dict["id"].append(submission.id)
			topics_dict["url"].append(submission.url)
			topics_dict["comms_num"].append(submission.num_comments)
			topics_dict["created"].append(submission.created)
			topics_dict["body"].append(submission.selftext)
			temp_array = []
			for comment in submission.comments.list():
				temp_array.append(comment)
				if comment.id not in conversedict:
					try:
						conversedict[comment.id] = [comment.body,comment.ups,comment.created,{}] # Original = [comment.body,{}]
						if comment.parent() != submission.id:
							parent = str(comment.parent())
							conversedict[parent][3][comment.id] = [comment.ups, comment.body, comment.created]
						#conversedict[comment.id] = [comment.body,{}]
						#if comment.parent() != submission.id:
					#		parent = str(comment.parent())
				#			pdb.set_trace()
			#				conversedict[parent][1][comment.id] = [comment.ups, comment.body]
					except:
						pass # Skip if no comments
						
			converse_df = pd.DataFrame(conversedict)
			count = count+1

			if save_comments == 1:
				converse_df.to_csv('%s' %(outname), index=False)


		# Convert Dictionary to Pandas Dataframe
		print("Creating Dataframe")
		topics_data = pd.DataFrame(topics_dict)

		# Convert Date to Timestamp
		_timestamp = topics_data["created"].apply(get_date)
		topics_data = topics_data.assign(timestamp = _timestamp)

		if save_posts == 1:
			print("Saving as csv")
			topics_data.to_csv('%sAllPosts' %(subreddit) + str(year) + '.csv', index=False) 

if get_comments_for_timeseries == 1:

	# Create Output Dictionary
	topics_dict = { "created":[], \
                "score":[], \
                "id":[], \
                "body": []}

	searches = ['IPCC','AR4','AR5'] # Kirilenko et al 2015 use climate change and global warming as search terms

	for search in searches:
		# Access Top x posts
		print("Retrieving Submissions")
		all_subreddit_comments = list(api.search_comments(q=search,after=start_epoch,subreddit=subreddit,filter=['url','author','title','subreddit'],limit=number_of_submissions))
		print("Appending Comments to Dataframe")
		count = 0
		num = len(all_subreddit_comments)
		for submission in all_subreddit_comments:
			print(str(count) + '/' + str(num))
			path = os.getcwd()
			dirname = path + '/Comments'
			if not os.path.exists(dirname):
				os.mkdir(dirname)
			outname = dirname + '/' + submission.id + '.csv'
			# Remove limit on comment extraction                
			topics_dict["created"].append(submission.created)
			topics_dict["score"].append(submission.score)
			topics_dict["id"].append(submission.id)
			topics_dict["body"].append(submission.body)
			count = count+1

	# Convert Dictionary to Pandas Dataframe
	print("Creating Dataframe")
	topics_data = pd.DataFrame(topics_dict)

	# Convert Date to Timestamp
	_timestamp = topics_data["created"].apply(get_date)
	topics_data = topics_data.assign(timestamp = _timestamp)

	# Export as CSV
	if save_posts == 1:
		print("Saving as csv")
		topics_data.to_csv('%s_IPCC_Comments.csv' %(subreddit), index=False)


if get_submissions_for_timeseries == 1:

	# Create Output Dictionary
	topics_dict = { "created":[], \
                "score":[], \
                "id":[], "url":[], \
                "comms_num": [], \
                "title": [], \
                "body":[]}

	searches = ['IPCC','AR4','AR5'] # Kirilenko et al 2015 use climate change and global warming as search terms

	for search in searches:
		# Access Top x posts
		print("Retrieving Submissions")
		all_subreddit = list(api.search_submissions(q=search,after=start_epoch,subreddit=subreddit,filter=['url','author','title','subreddit'],limit=number_of_submissions))

		print("Appending Submissions to Dataframe")
		count = 0
		num = len(all_subreddit)
		for submission in all_subreddit:
			print(str(count) + '/' + str(num))
			path = os.getcwd()
			dirname = path + '/Comments'
			if not os.path.exists(dirname):
				os.mkdir(dirname)
			outname = dirname + '/' + submission.id + '.csv'
			# Remove limit on comment extraction       
			topics_dict["created"].append(submission.created)         
			topics_dict["title"].append(submission.title)
			topics_dict["score"].append(submission.score)
			topics_dict["id"].append(submission.id)
			topics_dict["url"].append(submission.url)
			topics_dict["comms_num"].append(submission.num_comments)
			topics_dict["body"].append(submission.selftext)
			count = count+1

	# Convert Dictionary to Pandas Dataframe
	print("Creating Dataframe")
	topics_data = pd.DataFrame(topics_dict)

	# Convert Date to Timestamp
	_timestamp = topics_data["created"].apply(get_date)
	topics_data = topics_data.assign(timestamp = _timestamp)

	# Export as CSV
	if save_posts == 1:
		print("Saving as csv")
		topics_data.to_csv('%s_IPCC_Posts.csv' %(subreddit), index=False)