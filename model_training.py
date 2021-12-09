from influxdb import DataFrameClient
import numpy as np
import pandas as pd 
from datetime import datetime
from dotenv import load_dotenv
import os
import argparse
import matplotlib.pyplot as plt
import json

import mlflow
import mlflow.keras
from mlflow.models.signature import infer_signature

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

from pprint import pprint

from sklearn.model_selection import train_test_split

# Set numpy precision
np.set_printoptions(precision=3, suppress=True)

load_dotenv()

# Reference: https://github.com/arangoml/arangopipe/blob/master/arangopipe/arangopipe/arangopipe_analytics/rf_dataset_shift_detector.py
from sklearn.ensemble import RandomForestClassifier

def detect_dataset_shift(dataframe1, dataframe2):
	pd.options.mode.chained_assignment = None
	dataframe1.loc[:, "DS"] = 0
	dataframe2.loc[:, "DS"] = 1
	dfc = pd.concat([dataframe1, dataframe2])
	preds = dfc.columns.tolist()
	preds.remove("DS")
	X = dfc.loc[:, preds]
	Y = dfc.loc[:, "DS"]
	X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.33,random_state=42)
	clf = RandomForestClassifier(n_estimators=100,max_depth=3,random_state=0)
	clf.fit(X_train, y_train)
	acc_score = clf.score(X_test, y_test)
	return acc_score

HOST_URL_VAR = os.getenv('HOST_URL_VAR')
PORT_URL_VAR = os.getenv('PORT_URL_VAR')
INFLUXDB_DB = os.getenv('INFLUXDB_DB')
INFLUXDB_ADMIN_USER = os.getenv('INFLUXDB_ADMIN_USER')
INFLUXDB_ADMIN_PASSWORD = os.getenv('INFLUXDB_ADMIN_PASSWORD')
MQTT_CONSUMER_PROBE = os.getenv('MQTT_CONSUMER_PROBE')

current_path = os.getcwd()

parser = argparse.ArgumentParser(description="Send sensor & drift event messages")
parser.add_argument('--station_id', '-s', type=str.lower,
                    help="Station ID", default='1160629000')
parser.add_argument('--parameter_id', '-p', type=str.lower,
                    help="Parameter ID", default='112')
parser.add_argument('--network_parameters', '-n', type=str.lower,
                    help="Parameters for the LSTM model, e.g. 32,16,32", default='32,32,32')
parser.add_argument('--epochs', '-e', type=int,
                    help="Number of epochs for training LSTM model, e.g. 32,16,32", default=2)

args = parser.parse_args()

# station_id = '1160629000'
# parameter_id = '112'

query_where = f'SELECT * FROM {MQTT_CONSUMER_PROBE} WHERE time > now() - 3d and "topic"=\'sensor/{args.station_id}_{args.parameter_id}\';'


client = DataFrameClient(host=HOST_URL_VAR, port=PORT_URL_VAR, username=INFLUXDB_ADMIN_USER, password=INFLUXDB_ADMIN_PASSWORD, database=INFLUXDB_DB)

result = client.query(query_where)


bts_df = result[MQTT_CONSUMER_PROBE]

print("Result: \n{0}".format(bts_df))
print('Column names:',bts_df.columns)

mean_time = bts_df["unix_timestamp"].mean()
min_time = bts_df["unix_timestamp"].min()
bts_df["norm_time"] = (bts_df["unix_timestamp"]-mean_time)/(3600*1000000000)
bts_df = bts_df.sort_values(by=['norm_time'])
bts_df_grouped = bts_df.groupby(["station_id","parameter_id"])

bts_df_processed = bts_df.copy(deep=True)
mean_val = bts_df_processed['value'].mean()
bts_df_processed['norm_value'] = bts_df_processed['value']-mean_val
max_val = bts_df_processed['norm_value'].max()
bts_df_processed['norm_value'] = bts_df_processed['norm_value']/max_val
bts_df['actual_timestamp'] = bts_df['unix_timestamp'].apply(lambda x: datetime.utcfromtimestamp(x))

start_date_bts = str(bts_df['actual_timestamp'].min())
end_date_bts = str(bts_df['actual_timestamp'].max())
print('start_date_bts',start_date_bts)
print('end_date_bts',end_date_bts)
# print (bts_df.head())

# bts_df.to_csv(f'D:\\MasterDegree\\AdvancedTopicsInSoftwareSystems\\learning_examples\\influxdb-mqtt-playground\\output_drift\\{args.station_id}_{args.parameter_id}_bts.csv', )
bts_df.to_csv(os.path.join(current_path,'output_drift',f'{args.station_id}_{args.parameter_id}_bts.csv'),index=False)


# bts_df_processed.to_csv(f'D:\\MasterDegree\\AdvancedTopicsInSoftwareSystems\\learning_examples\\influxdb-mqtt-playground\\output_drift\\{args.station_id}_{args.parameter_id}_processed.csv', index=False)
bts_df_processed.to_csv(os.path.join(current_path,'output_drift',f'{args.station_id}_{args.parameter_id}_processed.csv'),index=False)

meta_df = pd.DataFrame(columns=['station_id', 'parameter_id', 'mean_val', 'max_val'])
for key,item in bts_df_grouped:
    sub_data = bts_df_grouped.get_group(key)
    mean_val = sub_data['value'].mean()
    sub_data['norm_value'] = sub_data['value']-mean_val
    max_val = sub_data['norm_value'].max()
    sub_data['norm_value'] = sub_data['norm_value']/max_val
    meta_df = meta_df.append({'station_id':(key[0]), 'parameter_id':key[1], 'mean_val':mean_val, 'max_val':max_val}, ignore_index=True)
    sub_data.sort_values(by=['norm_time']).to_csv("./output_drift/{}_{}.csv".format(str(int(key[0])),str(int(key[1]))), index=False)
    print("Finish: {}".format(key))
meta_df.to_csv("./output_drift/meta_data.csv", index=False)

columns = ['value','norm_time','norm_value']
bts_df_train = bts_df_processed[columns]
bts_df_train = bts_df_train.astype({'value':'float', 'norm_time':'float'})
bts_df_train = bts_df_train.reset_index(drop=True)
bts_df_train = bts_df_train.sort_values(by=['norm_time'])
print('bts_df_train',bts_df_train.head())
print('bts_df_train',bts_df_train.columns)

# Preparing data
bts_df_train = bts_df_train[:300]
serial_data = bts_df_train.drop(['value','norm_time'], axis=1)
serial_data['norm_1'] = serial_data['norm_value'].shift(1)
serial_data['norm_2'] = serial_data['norm_value'].shift(2)
serial_data['norm_3'] = serial_data['norm_value'].shift(3)
serial_data['norm_4'] = serial_data['norm_value'].shift(4)
serial_data['norm_5'] = serial_data['norm_value'].shift(5)
serial_data['norm_6'] = serial_data['norm_value'].shift(6)
serial_data = serial_data[6:]
print(serial_data)

# Split data into training and testing
train_dataset = serial_data.sample(frac=0.8, random_state=1)
test_dataset = serial_data.drop(train_dataset.index)

# Transform data into the pre-defined shape for training
train_features = np.array(train_dataset.drop(['norm_value'], axis=1))
train_features = np.array(train_features)[:,:,np.newaxis]
train_labels = np.array(train_dataset.drop(['norm_6'], axis=1))
train_labels = train_labels.reshape(train_labels.shape[0],train_labels.shape[1],1)

test_features = np.array(test_dataset.drop(['norm_value'], axis=1))
test_features = test_features.reshape(test_features.shape[0],test_features.shape[1],1)
test_labels = np.array(test_dataset.drop(['norm_6'], axis=1))
test_labels = test_labels.reshape(test_labels.shape[0],test_labels.shape[1],1)

RF_drift_score = detect_dataset_shift(train_dataset.drop(['norm_value'], axis=1), test_dataset.drop(['norm_value'], axis=1))
print('RF_drift_score',RF_drift_score)

print("Setting up model")

# # Define ML model
# ##################### ML Model ##############################
# model = keras.Sequential()
# model.add(layers.LSTM(32, return_sequences=True))
# model.add(layers.LSTM(32, return_sequences=True))
# model.add(layers.TimeDistributed(layers.Dense(1)))
# model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.005))
# #############################################################
# # Train the model
# print("Training model")
# model.fit(train_features, train_labels, epochs=200, batch_size=1, verbose=2)

# # Save model to file
# print("Saving model")
# model.save("./ml_model/lstm")

# # Convert to TFlite
# print("Converting model to tflite")
# converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model("./ml_model/lstm")
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.experimental_new_converter = True
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
# tflite_model = converter.convert()
# open("./ml_model/lstm/LSTM_single_series.tflite", "wb").write(tflite_model)
# print("Finish")




drift_objects = []
drift_detectors = ['adwin','hddm_a','kswin']

for drift_detector in drift_detectors:
	query_where = f'select * from {MQTT_CONSUMER_PROBE} where "topic"=\'sensor/{args.station_id}_{args.parameter_id}_{drift_detector}\';'
	print("Querying data: " + query_where)
	result = client.query(query_where)
	print("Result: \n{0}".format(result[MQTT_CONSUMER_PROBE]))
	df = result[MQTT_CONSUMER_PROBE]
	print('Column Names:',df.columns)
	# # In real case we don't need to remove index column
	# df.to_csv(f'D:\MasterDegree\AdvancedTopicsInSoftwareSystems\learning_examples\influxdb-mqtt-playground\output_drift\{drift_detector}_index.csv') 
	
	# In simulated, the actual timestamps should be used, so index timestamp column is ignored
	df = df.drop_duplicates()
	
	df['actual_timestamp'] = df['unix_timestamp'].apply(lambda x: datetime.utcfromtimestamp(x))
	df= df.sort_values(by='actual_timestamp')
	print('df.columns',df.columns)


	# df.groupby(f'drift_value_{drift_detector}').count().plot(kind='bar')
	ax = df[[f'drift_value_{drift_detector}']].hist()
	# fig = ax.get_figure()
	plt.title(f'Drift Histogram {drift_detector.upper()} Method')
	plt.xlabel('Drift Values')
	plt.ylabel('Count')
	# plt.savefig(f'D:\\MasterDegree\\AdvancedTopicsInSoftwareSystems\\learning_examples\\influxdb-mqtt-playground\\output_drift\\figures\\{drift_detector}.png')
	plt.savefig(os.path.join(current_path,'output_drift','figures',f'{drift_detector}.png'))

	# df.to_csv(f'D:\\MasterDegree\\AdvancedTopicsInSoftwareSystems\\learning_examples\\influxdb-mqtt-playground\\output_drift\\{drift_detector}.csv', index=False)
	# df.to_csv(os.path.join(current_path,'output_drift',f'{drift_detector}.csv'))

	dict_values = df[f'drift_value_{drift_detector}'].value_counts().to_dict()
	dict_values = {str(key): value for key, value in dict_values.items()}

	drift_object = {
		'drift_method': drift_detector,
		'total_drifts': df.shape[0],
		'start_date': str(df['actual_timestamp'].iloc[0]),
		'end_date': str(df['actual_timestamp'].iloc[-1]),
		'max_value_drift': df[f'drift_value_{drift_detector}'].max(),
		'min_value_drift': df[f'drift_value_{drift_detector}'].min(),
		'drift_values': dict_values,
	}
	# print('drift_object',drift_object)
	# pprint(drift_object)
	drift_objects.append(drift_object)

print('drift_objects')
pprint(drift_objects)


with mlflow.start_run():
	# Define ML model
	##################### ML Model ##############################
	model = keras.Sequential()
	node_param = args.network_parameters.split(',')
	print(node_param)
	for i in range(len(node_param)):
		model.add(layers.LSTM(int(node_param[i]), return_sequences=True))
	# model.add(layers.LSTM(32, return_sequences=True))
	# model.add(layers.LSTM(32, return_sequences=True))
	model.add(layers.TimeDistributed(layers.Dense(1)))
	model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.005))
	#############################################################

	fitted_model = model.fit(train_features, train_labels, epochs=args.epochs, batch_size=1, verbose=2, validation_data=(test_features, test_labels))
	signature = infer_signature(test_features, model.predict(test_features))

	# import shap
	# # Use the training data for deep explainer => can use fewer instances
	# explainer = shap.DeepExplainer(model, train_features)
	# # explaining each prediction requires 2 * background dataset size runs
	# shap_values = explainer.shap_values(test_features)
	# print('shap_values',shap_values)

	# # init the JS visualization code
	# shap.initjs()
	# shap.force_plot(explainer.expected_value[0], shap_values[0][0], features)

	# Log params
	mlflow.log_param("Number of layer", len(node_param))
	mlflow.log_param("Number of node each layer", node_param)
	mlflow.log_param("Number of epochs", args.epochs)
	fit_history = fitted_model.history
	# Log metrics
	for key in fit_history:
	    mlflow.log_metric(key, fit_history[key][-1])
	# Log artifacts
	for drift_detector in drift_detectors:
		mlflow.log_artifact(f"./output_drift/figures/{drift_detector}.png")

	ruuid = mlflow.active_run().info.run_uuid
	meta_object = {
		'station_id': args.station_id,
		'start_date': start_date_bts,
		'end_date': end_date_bts,
		'timestamp': str(datetime.now()),
		'station_id': args.station_id,
		'parameter_id': args.parameter_id,
		'experiment_id': ruuid,
		'mean_val': bts_df_processed['value'].mean(),
		'max_val': bts_df_processed['norm_value'].max(),
		'drift_objects':drift_objects,
		'dataset_drift_score': RF_drift_score
	}
	print(meta_object)
	with open("./output_drift/json/metadata.json", "w+") as f:
		f.write(json.dumps(meta_object, indent=4))

	mlflow.log_artifact("./output_drift/json/metadata.json")

	# # Save model to file
	# print("Saving model")
	# model.save("./ml_model/lstm")

	# # Convert to TFlite
	# print("Converting model to tflite")
	# converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model("./ml_model/lstm")
	# converter.optimizations = [tf.lite.Optimize.DEFAULT]
	# converter.experimental_new_converter = True
	# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
	# tflite_model = converter.convert()
	# open("./ml_model/lstm/LSTM_single_series.tflite", "wb").write(tflite_model)
	# print("Finish")

	# Create an input example to store in the MLflow model registry
	input_example = np.expand_dims(train_features[0], axis=0)

	# Let's log the model in the MLflow model registry
	model_name = 'LSTM_model'
	mlflow.keras.log_model(model,"LSTM_model", signature=signature, input_example=input_example)


