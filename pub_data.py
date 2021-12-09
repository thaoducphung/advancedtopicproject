

import paho.mqtt.client as mqtt
import random
import time
import json
import pprint

import pandas as pd
import argparse

from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.drift_detection.hddm_a import HDDM_A
from skmultiflow.drift_detection import PageHinkley
from skmultiflow.drift_detection import KSWIN

from datetime import datetime

broker = "localhost"
port = 1883

# publish callback function
def on_publish(client, userdata, result):
    print("data published \n")
    pass

parser = argparse.ArgumentParser(description="Send sensor & drift event messages")
parser.add_argument('--station_id', '-s', type=str.lower,
                    help="Station ID", default='1160629000')
parser.add_argument('--parameter_id', '-p', type=str.lower,
                    help="Parameter ID", default='112')
args = parser.parse_args()



data_path = f"D:\\MasterDegree\\AdvancedTopicsInSoftwareSystems\\learning_examples\\influxdb-mqtt-playground\\raw_data\\{args.station_id}_{args.parameter_id}.csv"
dataframe = pd.read_csv(data_path)
dataframe['datetime_timestamp'] = pd.to_datetime(dataframe['reading_time'], format='%Y-%m-%d %H:%M:%S', errors='ignore')
dataframe['unix_timestamp'] = dataframe['datetime_timestamp'].apply(lambda x: x.timestamp())
# dataframe['actual_timestamp'] = dataframe['unix_timestamp'].apply(lambda x: datetime.utcfromtimestamp(x))
print(dataframe.head())

# create client object and connect
client = mqtt.Client()
client.username_pw_set("rabbitmq", password='rabbitmq')
client.connect(broker, port, 45)

# assign publish callback function
client.on_publish = on_publish

adwin_detector = ADWIN()
pagehinkley_detector = PageHinkley()
hddm_a_detector = HDDM_A()
kswin_detector = KSWIN(alpha=0.01)

# publish messages
# while True:
    
for index, row in dataframe.iterrows():
    
    adwin_detector.add_element(row['value'])
    # pagehinkley_detector.add_element(row['value'])
    kswin_detector.add_element(row['value'])
    hddm_a_detector.add_element(row['value'])

    dict_msg = {
        "id": row['id'],
        "unix_timestamp": row['unix_timestamp'],
        "station_id": row['station_id'],
        "parameter_id": row['parameter_id'],
        "value": row['value'],
    }
    msg = json.dumps(dict_msg)
    ret = client.publish(f"sensor/{args.station_id}_{args.parameter_id}", msg)
    print(msg)

    if adwin_detector.detected_change():
        print('Drift method: ADWIN\t-\tChange detected in data:' + str(row['value']) + '\tat index: ' + str(index) + 'in' + str(row['unix_timestamp']))
        drift_event =  {
          "id" : row['id'],  
          "drift_value_adwin"  : row['value'],
          "station_id" : row['station_id'], 
          "parameter_id" : row['parameter_id'], 
          "unix_timestamp" : row['unix_timestamp']
        }
        msg = json.dumps(drift_event)
        ret = client.publish(f"sensor/{args.station_id}_{args.parameter_id}_adwin", msg)

    # if pagehinkley_detector.detected_change():
    #     print('Drift method: PageHinkley\t-\tChange detected in data:' + str(row['value']) + '\tat index: ' + str(index) + 'in' + str(row['unix_timestamp']))
    #     drift_event =  {
    #       "id" : row['id'],  
    #       "drift_value"  : row['value'],
    #       "station_id" : row['station_id'], 
    #       "parameter_id" : row['parameter_id'], 
    #       "unix_timestamp" : row['unix_timestamp']
    #     }
    #     ret = client.publish(f"sensor/{args.station_id}_{args.parameter_id}_pagehinkley", drift_event)

    if hddm_a_detector.detected_change():
        print('Drift method: HDDM_A\t-\tChange detected in data:' + str(row['value']) + '\tat index: ' + str(index) + 'in' + str(row['unix_timestamp']))
        drift_event =  {
          "id" : row['id'],  
          "drift_value_hddm_a"  : row['value'],
          "station_id" : row['station_id'], 
          "parameter_id" : row['parameter_id'], 
          "unix_timestamp" : row['unix_timestamp']
        }
        msg = json.dumps(drift_event)
        ret = client.publish(f"sensor/{args.station_id}_{args.parameter_id}_hddm_a", msg)

    if kswin_detector.detected_change():
        print('Drift method: KSWIN\tChange detected in data: ' + str(row['value']) + '\tat index: ' + str(index) + 'in' + str(row['unix_timestamp']))
        drift_event =  {
          "id" : row['id'],  
          "drift_value_kswin"  : row['value'],
          "station_id" : row['station_id'], 
          "parameter_id" : row['parameter_id'], 
          "unix_timestamp" : row['unix_timestamp']
        }
        msg = json.dumps(drift_event)
        ret = client.publish(f"sensor/{args.station_id}_{args.parameter_id}_kswin", msg)

    time.sleep(0.2)