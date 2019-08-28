'''
Extract users from a specified subreddit

Author: Josh Williams
Date: 17/07/2019
Update: 17/07/2019
'''

# Import Modules
import praw
from psaw import PushshiftAPI
import pandas as pd
import datetime as dt
import os
import time
start = time.time()

# Options
save_users = 1
save_comments = 1
get_top_submissions = 0
get_year_submissions = 1

# All Posts
number_of_submissions = None # Set number of posts (None = all posts)

# Top Posts
top_post_limit = 20

# Create Functions
def get_date(created):
    return dt.datetime.fromtimestamp(created)

# Set up Reddit API
reddit = praw.Reddit(client_id='INSERT_CLIENT_ID_HERE', \
                     client_secret='INSERT_CLIENT_SECRET_HERE', \
                     user_agent='INSERT_USER_AGENT_HERE', \
                     username='INSERT_USERNAME_HERE', \
                     password='INSERT_PASSWORD_HERE')

api = PushshiftAPI(reddit) # Use Pushshift API to get around 1000 submission limit imposed by praw

# Access Climate Skepticism Subreddit
subreddit = reddit.subreddit('climateskeptics')

# Loop through top submissions and append to output dataframe
if get_top_submissions == 1:
	# Create Output Dictionary
	topics_dict = { "title":[], \
				"author":[], \
                "score":[], \
                "id":[], \
                "created": []}

	# Access Top x posts
	print("Retrieving Submissions")
	submissions = subreddit.top(limit=top_post_limit)

	print("Appending Submissions to Dataframe")
	count = 0
	for submission in submissions:
		print(count)
		path = os.getcwd()
		conversedict = {}
		dirname = path + '/Users'
		if not os.path.exists(dirname):
			os.mkdir(dirname)
		outname = dirname + '/' + submission.id + '.csv'
		# Remove limit on comment extraction                
		submission.comments.replace_more(limit=None)
		try:
			topics_dict["title"].append(submission.title)
			topics_dict["author"].append(submission.author.name)
			topics_dict["score"].append(submission.score)
			topics_dict["id"].append(submission.id)
			topics_dict["created"].append(submission.created)
		except:
			pass # Skip where error, ie account deleted
		temp_array = []
		for comment in submission.comments.list():
			temp_array.append(comment)
			if comment.id not in conversedict:
				try:
					conversedict[comment.id] = [comment.author.name,comment.ups,{}] # Original = [comment.body,{}]
					if comment.parent() != submission.id:
						parent = str(comment.parent())
						try:
							conversedict[parent][2][comment.id] = [comment.ups, comment.author.name]
						except:
							pass # Pass by deleted comments
				except:
					pass # Pass by deleted comments#

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
	if save_users == 1:
		print("Saving as csv")
		topics_data.to_csv('%sTop%sPostsUsers.csv' %(subreddit, str(top_post_limit)), index=False) 

# Loop through top submissions and append to output dataframe
if get_year_submissions == 1:
	years=[2019]
	total_posts = []
	for year in years:
		print('Getting Submissions for %s' %(year))
		start_epoch = int(dt.datetime(year, 1, 1).timestamp()) # Set start point for post extraction
		end_epoch = int(dt.datetime(year,12,31).timestamp()) # Set end point

		# Create Output Dictionary
		topics_dict = { "title":[], \
					"author":[], \
	                "score":[], \
	                "id":[], \
	                "created": []}

		# Access Top x posts
		print("Retrieving Submissions")
		submissions = list(api.search_submissions(before=end_epoch,after=start_epoch,subreddit=subreddit,filter=['url','author','title','subreddit'],limit=number_of_submissions))
		total_posts.append(len(submissions))
		print(len(submissions))
		print("Appending Submissions to Dataframe")
		count = 1
		for submission in submissions:
			print(str(count) + '/' + str(len(submissions)))
			path = os.getcwd()
			conversedict = {}
			dirname = path + '/Users'
			if not os.path.exists(dirname):
				os.mkdir(dirname)
			outname = dirname + '/' + submission.id + '.csv'
			# Remove limit on comment extraction                
			submission.comments.replace_more(limit=None)
			try:
				topics_dict["author"].append(submission.author.name)
				topics_dict["title"].append(submission.title)
				topics_dict["score"].append(submission.score)
				topics_dict["id"].append(submission.id)
				topics_dict["created"].append(submission.created)
			except:
				pass # Skip where error, ie account deleted
			temp_array = []
			for comment in submission.comments.list():
				temp_array.append(comment)
				if comment.id not in conversedict:
					try:
						conversedict[comment.id] = [comment.author.name,comment.ups,{}] # Original = [comment.body,{}]
						if comment.parent() != submission.id:
							parent = str(comment.parent())
							try:
								conversedict[parent][2][comment.id] = [comment.ups, comment.author.name]
							except:
								pass # Pass by deleted comments
					except:
						pass # Pass by deleted comments#

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
		if save_users == 1:
			print("Saving as csv")
			topics_data.to_csv('%sAllUsers' %(subreddit) + str(year) + '.csv', index=False) 

end = time.time()
time_taken = end - start
time_taken = time.strftime('%H:%M:%S', time.gmtime(time_taken))
print('Time Taken: ' + time_taken)