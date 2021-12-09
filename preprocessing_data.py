import os
import pandas as pd
import glob


# Loading dataset from csv file
dir_path = os.path.dirname(os.path.realpath(__file__))
results = glob.glob(os.path.join(dir_path,'raw_data/*.csv' ))
print(results)


dataset = pd.read_csv(os.path.join(dir_path,'./raw_data/param-2017-04-18-00-vn.csv'))
print(dataset.head())
print()

# Explore the data by:
## Checking unique values from each column
## Check data types of each column
## Check number of missing values in each column
def check_dataframe(dataframe):
	df = pd.DataFrame()
	df["Missing values"] = dataframe.isnull().sum()
	df["Unique values"] = dataframe.nunique()
	df["Dtypes"] = dataframe.dtypes
	return df 

check_info = check_dataframe(dataset)
print(check_info)
print()

# Describe data
print(dataset.describe())
print()

# Check column name:
def check_column_name(dataset):
	for column in dataset.columns:
		print('Column Name:',column)
	print()
check_column_name(dataset)

# Interest columns
columns = ['parameter_id','station_id']

# Print unique columns for columns (station_id and parameter_id)
for column in columns:
	print('Column Name:',column)
	print('\t + Unique Value:', dataset[column].unique())
	print()

for each_file in results:
	print('File name:',each_file)
	dataset = pd.read_csv(each_file)
	# For each unique station value
	for station_id in dataset['station_id'].unique():
		# print('Station ID:', station_id)
		# Iterate through every parameter/sensor id to get data
		for parameter_id in dataset['parameter_id'].unique():
			# print('parameter_id',parameter_id)
			station_parameter_id = dataset[(dataset.station_id==station_id) & (dataset.parameter_id==parameter_id)].copy()
			current_dir = os.path.join(os.getcwd(),"raw_data")
			current_dir = os.path.join(current_dir,"process_data")
			if not os.path.exists(current_dir):
				os.mkdir(current_dir)
			station_dir = os.path.join(current_dir,str(station_id))

			if not os.path.exists(station_dir):
				os.mkdir(station_dir)
			parameter_dir = os.path.join(station_dir,str(parameter_id))
			if not os.path.exists(parameter_dir):
				os.mkdir(parameter_dir)

			station_parameter_id['reading_time'] = pd.to_datetime(station_parameter_id['reading_time'], format='%Y-%m-%d %I:%M:%S %p', errors='ignore')
			station_parameter_id = station_parameter_id.sort_values('reading_time')

			if os.path.exists(os.path.join(parameter_dir,str(parameter_id)+'.csv')):
				station_parameter_id.to_csv(os.path.join(parameter_dir,str(parameter_id)+'.csv'),index=False, mode='a', header=False)
			else:
				station_parameter_id.to_csv(os.path.join(parameter_dir,str(parameter_id)+'.csv'),index=False, mode='a')



		# print(station_parameter_id.head())
		# print(station_parameter_id.info())
	# stop_here_for_only_1_station


