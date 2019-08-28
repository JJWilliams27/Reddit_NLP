# Reddit_NLP
Natural Language Processing of Reddit Big Data using Python

These scripts are set up to replicate the methodology used in the following report: 

As such, to use them for alternative studies, some manual work is required to edit the scripts (i.e. output filenames, etc), especially when plotting data.

## Usage 1 : NLP of Reddit Data

### Scraping Posts and Comments
Edit *scrape_reddit.py*, and insert the necessary information to use the API (username, password, client_id, client_secret, user_agent). There are six options at the top of the code. I recommend leaving *save_posts* and *save_comments* equal to 1, such that the scraped data is saved as csvs, usable with subsequent code. You have a choice as to whether to scrape the top rated submissions (posts + comments) for your subreddit of choice, or all submissions within a specified time period (or both). There is also the option to search for specific keywords within submissions of a subreddit to produce a time series.

Set *subreddit* to your subreddit of choice. If you wish to use multiple subreddits, you will have to do them one at a time, or add a loop to the code. *Number_of_posts* should be set to the maximum number of posts to scrape, or to *None* if you wish to scrape everything (i.e. no limit). Note here that if using *get_top_submissions*, there is a limit set by the API (whereas get_all_submissions uses psaw which circumvents this), so don't set this beyond ~1000 or so. *Start_epoch* is a datetime object designating the start time of your search. Slightly confusingly, the version of this at the top of the code only relates to the keyword timeseries options.

If using *get_all_submissions*, scroll down to the respective loop. Here, you will see an array titled *years*. In here, enter all of the years for which you wish to scrape data.

If using a keyword search (get_submissions_for_timeseries), scroll down to the respective loop. There is an array titled *searches*. In here, enter the words you wish to search for. You may also want to change the output file name to reflect your search.

Now, run the script. It will output the relevant posts in a csv starting with *subredditname*. Associated comments will be stored in a subdirectory named *Comments*.

If you have run the script for multiple years, you should use *join_yearly_dataframes.py* to combine them into one file. Edit this script, and set the *years* array to the years that you have used. Set *run_type* to *Submissions*, *df* to read your data csv, and the last line to an output filename of your choice.

### Plotting a Time Series
If you have scraped data to plot a time series, follow *plot_time_series_subplot.py* to produce a figure displaying post/comment frequency over time (aggregated into monthly bins). This will require considerable editing if not re-producing the results from the above paper.

### Pre-Processing the Text Data 
Edit *pre_processing.py*, and change *data* to the csv containing your scraped post data (*subredditnameAllPosts.csv*). If you wish to compute bigrams, scroll to the very bottom and set *use_bigrams* to 1 (default 0).

Run the script. This is quite process-intensive for large datasets (I had to leave this running overnight when using an entire subreddit archive of data). This will pre-process the text data, outputting csvs ending with *_PP* (denoting pre-processed) for posts and comments. This script will also output csvs titled *LinkedSubreddits.csv*, *LinkedURLs.csv* and *LinkedURL_subreddits.csv*, which contain data on the number of links to other subreddits (in both /r/subreddit and www.reddit.com/r/subreddit format) and URLs within the dataset. If you add a *plt.show()* to the end of the script, these will be plotted here, but there are other scripts in this repository to plot this data outside of this script (*plot_linked_subreddits.py*, *plot_linked_url_subreddits.py, *plot_linked_urls.py*).

Finally, dicitonary, corpora and sent files are saved, which are used in the LDA modelling. Note that with a large dataset (ie a full subreddit archive), the latter two can be quite large files.

### LDA Topic Modelling
Edit *process_text.py*. Scroll to the very bottom. If you wish to test a number of LDA models (i.e. with different numbers of topics), set *compute_coherence* to 1. If not, set *number_of_topics* to the number of topics you wish to model. Finally, set *num_topics_to_analyse* to the number of posts/comments you wish to analyse for each topic. For example, setting this to 50 will output a csv with the 50 most related posts/comments for each topic, allowing you to go through them and aid in understanding the topic.

If testing multiple models (*compute_coherence = 1*), scroll back to the top. On lines 55-57 are three further options. *Min_topics* is the minimum number of topics to be modelled, *max_topics* is the maximum number (though this should be set to maximum number + topics step given how range works in Python), and *topics_step* is the step size (i.e. min=2, max=12, step=2 will test models with 2,4,6,8 and 10 topics). You can now run the script.

Note that the processing can be extremely intensive and take a long time. Also, the processing will output a number of very large files, and so I would recommened running from a directory/external device with enough space (on the order of 10s of Gbs if using a full subreddit archive - I used 15 models for 1 subreddit, and the required memory was about 18Gb).

For each model, the script will output the following:
* *topic_timeseries_numtopics.csv* : Used for creating a timeseries for each topic.
* *topic_naming_numtopics.csv* : Used to aid in naming each topic (contains the X most related posts/comments for each topic, set by *num_topics_to_analyse*.
* *topic_analyses_numtopics.csv* : Generic analyses for each topic (i.e. keywords, percentage of data, mean score, length, sentiment)
* *final_dataframe_numtopics.csv* : Final dataframe following all processing.

In addition, it will produce *LDA_Model_Performance.csv*, which allows for the assessment of each model.

If only running the model once, the above will be produced, but named as *topic_timeseries.csv*, *topic_naming.csv*, *topic_analyses.csv* and *final_dataframe.csv*.

At the end of processing, the code will output counts of the occurances of "climate change" and "global warming" in the data. This can be ignored or removed if not of interest. Further, plots of the Top20 most frequent tokens and lemma are produced. These, and the count of climate change vs global warming, can be re-produced quickly by using *short_process.py* once *process_text.py* has been run once.

To assess the optimum number of topics, you can assess the coherence. Use *plot_lda_coherence.py* to produce a plot of model coherence vs number of topics. 

### Additional Plots
Note that if not re-producing the data in the associated paper, these scripts will require considerable editing.

Use *corpus_plots.py* to produce the following: Plots of the top 20 lemma for each subreddit, and a boxplot showing ranges in sentiment, score, and length for each modelled topic.

Use *plot_sentiment.py* to produce a plot of sentiment distribution.

## Usage 2: Creating a Subreddit Network

### Scraping Reddit Users
As reddit user data is anomalous, it is not as simple as accessing a user's profile to see the subreddits in which they are active. As such, we must trawl through submissions within a subreddit, and get the username of the associated users.

Edit *scrape_reddit_users.py*. Add the necessary data to use the API in lines 32-36.

Leave *save_posts* and *save_comments* as equal to 1 if you wish to save the outputs (note these are different to those of *scrape_reddit.py*, as they include the author data). A subfolder called *Users* will be created in which these will be saved. Next, set *subreddit* to your subreddit of choice.

If you only wish to extract user data from the top posts, set *get_top_submissions* to 1. Set *top_post_limit* to the number of posts - reddit API has a limit of 1000.

Alternatively, if you wish to extract user data from all submissions, set *get_all_submissions* to 1. Scroll down to line 114, and set the *years* array to all of the years for which you wish to get data. An output csv will be created for each year.

Run the script. If you have extracted data from multiple years, use *join_yearly_dataframes.py* to combine them into one csv. You will need to edit the script and set *run_type* to *Users*, and set the input data sources as your files.

### Processing User Data
Edit *process_reddit_users.py*, and as before, insert the relevant data to use the Reddit API. I would recommend leaving *save_users* as 1 such that the data is saved as a csv (*subredditnameAllUsersProcessed.csv*). Set your input subreddit as *input_subreddit* - you can use multiple subreddits here if you wish.

There are 6 options. *start_epoch* and *end_epoch* are the start and end dates (as datetime objects) between which submissions of a given user are searched for (i.e. if set to 2018 and 2019, for each user, the script will assess their posts and comments between 2018 and 2019). *Number_of_posts* and *number_of_comments* is the number of posts/comments that will be assessed for each user - note that setting these to high values (or None for no limit) will take an extremely long time to process. *Min_submissions* and *min_submissions_to_sr* are filtering options. The former filters out users who have submitted less than X times to your input subreddit. The latter looks at the other subreddits to which a user has submitted, and adds those which have been submitted to more than X times to the output dataframe.

Run the script. It may take a while, and will output a file named *subredditname_SubredditNetwork_FULL.csv*.

### Creating a Network Diagram
Edit *create_subreddit_network.py*. Change *input_subreddits* to your subreddit(s) of analysis. 

There are two options. *Min_users* refers to the minimum number of distinct users required for a subreddit to be added to the network. *Min_submissions* refers to the minimum number of submissions by users of your input subreddit(s) within another subreddit for this other subreddit to be added to the network. 

Change your input filenames in lines 23 and 24. There may be considerable editing required if you wish to use only one subreddit, or more than two.

Run the script, and a network diagram will be produced. The final two lines (commented out by default) can be uncommented if you wish to output the network in gephi format.




