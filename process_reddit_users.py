import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import ast
import collections
import praw
from psaw import PushshiftAPI
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

save_users = 1

start_epoch = int(dt.datetime(2008, 1, 1).timestamp()) 
end_epoch = int(dt.datetime(2019, 12, 31).timestamp()) 
number_of_posts = 500 # Number of posts to find for each user, None = no limit. This is sorted by score (i.e. setting this to 10 will get the top 10 highest scored posts)
number_of_comments = 500 # Number of comments to find for each user, None = no limit. This is sorted by score (i.e. setting this to 10 will get the top 10 highest scored comments)
min_submissions = 10 # Minimum submissions in input subreddit for a user to be considered
min_submissions_to_sr = 10 # Minimum submissions by a user in other subreddits for those subreddits to be considered

# Set up Reddit API
reddit = praw.Reddit(client_id='INSERT_CLIENT_ID_HERE', \
                     client_secret='INSERT_CLIENT_SECRET_HERE', \
                     user_agent='INSERT_USER_AGENT_HERE', \
                     username='INSERT_USERNAME_HERE', \
                     password='INSERT_PASSWORD_HERE')

api = PushshiftAPI(reddit) # Use Pushshift API to get around 1000 submission limit imposed by praw

input_subreddit = ['climateskeptics'] # Make sure this follows the URL (i.e. /r/climateskeptics, therefore lowercase)

def main():
	for in_sr in input_subreddit:
		print('Processing Author Data')
		# Read in CSV as pandas dataframe
		data = pd.read_csv('%sAllUsers.csv' %(in_sr))

		# Read in CSV as pandas dataframe
		toplevelcomments = pd.DataFrame(columns=['Author','Score'])
		for i in tqdm(list(range(0,len(data['id'])))):
			if os.path.isfile('Users/%s.csv' %(data['id'][i])):
				try:
					data2 = pd.read_csv('Users/' + '%s.csv' %(data['id'][i]))

					row = next(data2.iterrows())
					tempdf = pd.DataFrame(row[1])
					tempdf['Score'] = data2.iloc[1]
					tempdf.columns = ['Author','Score']
					toplevelcomments = pd.concat([toplevelcomments,tempdf])

					# Subcomments
					subcom_list = []
					scores = []
					for j in list(range(0,data2.shape[1])):
						try:
							subc = ast.literal_eval(data2.loc[2,:][j]) # Read text representation of dictionary as dictionary
						except:
							print(i)
							print(data2.loc[2,:][j])
						for key in subc:
							try:
								val = subc.get(key)
								score = val[0]
								author = val[1]
								subcom_list.append(author)
								scores.append(score)
							except:
								pass
					if len(subcom_list)>0:
						try:
							tempsc = pd.DataFrame(subcom_list)
							tempsc.columns = ['Author']
							tempsc['Score'] = scores
							tempsc = tempsc[tempsc['Author'].map(lambda d: len(d)) > 0]
							subcomments = pd.concat([subcomments,tempsc])
						except NameError:
							subcomments = pd.DataFrame(subcom_list)
							subcomments.columns = ['Author']
							subcomments['Score'] = scores
							subcomments = subcomments[subcomments['Author'].map(lambda d: len(d)) > 0]
				except:
					pass # Pass if no comments for post	

		posts = pd.DataFrame()
		posts['Author'] = data['author']
		posts['Score'] = data['score']

		# Make sure column names are the same
		posts.columns = ['Author','Score']
		toplevelcomments.columns = ['Author','Score']
		subcomments.columns = ['Author','Score']

		full_df = pd.concat([posts,toplevelcomments,subcomments])
		full_df.reset_index(drop=True, inplace=True)

		full_df_grp = full_df.groupby('Author')

		authors = []
		author_comms = []
		author_scores = []
		for i, grp in full_df_grp:
			authors.append(i)
			author_comms.append(len(grp))
			temp = np.array(grp['Score'])
			author_scores.append(np.sum(temp.astype(np.float))) # Calculate Total Sum of Post/Comment Votes

		auth_df = pd.DataFrame()
		auth_df['Author'] = authors
		auth_df['Num_Comments'] = author_comms
		auth_df['Total_Score'] = author_scores

		auth_df = auth_df.sort_values(by=['Num_Comments'],ascending=False)
		auth_df = auth_df[auth_df['Num_Comments'] > min_submissions]
		auth_df.reset_index(drop=True, inplace=True)
		print("Saving Users")
		if save_users == 1:
			auth_df.to_csv('%sAllUsersProcessed.csv' %(in_sr))

		#user = reddit.redditor(auth_df['Author'][3]) # Get user
		#tt = list(user.submissions.new()) # Get user submissions
		#tt[0].subreddit # get subreddit of submission

		authors = list(auth_df['Author'])
		sr_df = pd.DataFrame(columns=['Subreddits','Total_Submissions','Distinct_Users'])

		print('Getting User Submissions in other Subreddits')
		for auth in tqdm(authors):
			try:
				user = reddit.redditor(auth)
				#subs = list(user.submissions.new()) # Get 100 latest submissions 
				subs = list(api.search_submissions(author=user.name,after=start_epoch,before=end_epoch,sort_type="score",limit=number_of_posts)) # Get all submissions for given time period
				subs.extend(api.search_comments(author=user.name,after=start_epoch,before=end_epoch,sort_type="score",limit=number_of_comments))
				srs = [] # list for subreddits
				for sub in subs:
					sr = sub.subreddit.display_name
					# Create list of subreddits, count occurances to get submission numbers
					srs.append(sr)

				counter=collections.Counter(srs) # Get submissions per subreddit
				counter=dict(counter) # Convert to dictionary for looping
				srs = list(set(srs)) # Convert to set to just get unique values, then convert back to list for looping
				for sr in srs:
					num_subs = counter.get(sr) # Loop through subreddit list - can do this as subreddit names are also the keys in the dictionary
					if num_subs > min_submissions_to_sr: # If enough submissions in subreddit, add to output
						if sr not in sr_df['Subreddits'].unique():
							sr_df.loc[len(sr_df)] = [sr] + ([num_subs,1]) # If subreddit doesnt exist in the dataframe, add a new row with subreddit name, number of submissions and a value of 1 for distinct users
						else:
							idx = sr_df.loc[sr_df['Subreddits'] == sr].index[0]
							new_subs = sr_df['Total_Submissions'][idx] + num_subs
							new_users = sr_df['Distinct_Users'][idx] + 1
							sr_df.set_value(idx,'Total_Submissions',new_subs)
							sr_df.set_value(idx,'Distinct_Users',new_users)
			except:
				pass # Pass by deleted accounts

		sr_df = sr_df.sort_values(by=['Total_Submissions'],ascending=False)
		sr_df2 = sr_df.sort_values(by=['Distinct_Users'],ascending=False)
		print('...')
		print('...')

		# Remove input subreddit from dataframe (i.e. want to look at connections to subreddits from the input subreddit)
		idx = sr_df.loc[sr_df['Subreddits'] == in_sr].index[0]
		sr_df_final = sr_df.drop(idx) 

		# Save
		print('Saving Final Dataframe as CSV')
		sr_df_final.to_csv('%s_SubredditNetwork_FULL.csv' %(in_sr))


if __name__ == "__main__":
	main()