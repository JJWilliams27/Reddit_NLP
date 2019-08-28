# Combine Yearly Post Dataframes

# Import Modules
import pandas as pd
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)

years = [2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]

run_type = 'Users' # 'Users' or 'Submissions'

# Read first year 
df = pd.read_csv('ClimateSkepticsAllUsers2008.csv')

# Loop through years and concatenate

for i in list(range(1,len(years))):
	year = str(years[i])
	df_temp = pd.read_csv('ClimateSkepticsAllUsers%s.csv' %(year))

	df = pd.concat([df,df_temp])

# Calculate total comments
if run_type == 'Submissions':
	num_comms = df['comms_num'].sum()
	print('Total Posts: %s' %(str(len(df))))
	print('Total Comments: %s' %(str(num_comms)))

# Save full DataFrame
print("Saving Full DataFrame")
df.to_csv('ClimateSkepticsAllUsers.csv')