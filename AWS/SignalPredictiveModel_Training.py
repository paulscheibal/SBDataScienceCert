# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 08:48:45 2020

@author: Paul Scheibal
"""
#
#    This program reads in accelerometer data from multiple files
#    which comes from ball bearing vibration.  The program 
#    classifies the data as follows:
#        
#        0  Baseline Data - 0 HP workload normal, class 0
#        1  Baseline Data - 1 HP workload normal, class 0
#        2  Baseline Data - 2 HP workload normal, class 0
#        3  Baseline Data - 3 HP workload normal, class 0
#        4  Faulty Data -   0 HP workload .007 inches EDM, class 1
#        5  Faulty Data -   0 HP workload .014 inches EDM, class 2 
#        6  Faulty Data -   0 HP workload .021 inches EDM, class 3
#        7  Faulty Data -   0 HP workload .028 inches EDM, class 4 
#        8  Faulty Data -   1 HP workload .007 inches EDM, class 1
#        9  Faulty Data -   1 HP workload .014 inches EDM, class 2  
#        10 Faulty Data -   1 HP workload .021 inches EDM, class 3
#        11 Faulty Data -   1 HP workload .028 inches EDM, class 4  
#        12 Faulty Data -   2 HP workload .007 inches EDM, class 1
#        13 Faulty Data -   2 HP workload .014 inches EDM, class 2  
#        14 Faulty Data -   2 HP workload .021 inches EDM, class 3 
#        15 Faulty Data -   2 HP workload .028 inches EDM, class 4 
#        16 Faulty Data -   3 HP workload .007 inches EDM, class 1
#        17 Faulty Data -   3 HP workload .014 inches EDM, class 2  
#        18 Faulty Data -   3 HP workload .021 inches EDM, class 3 
#        19 Faulty Data -   3 HP workload .028 inches EDM, class 4 
#        
#        EDM = electromagnetic machining introduced defect
#
#        The accelerometer data is sampled at 12,000 samples per second.
#
# This program uses 1D CNN machine learning using raw signal data as input

#
#  This simulates the generation of sensor data and predicts signal defects
#  0 - segment is normal operation
#  1 - segment is .007 defect of ball bearing - monitoring for further deterioration
#  2 - segment is .014 defect of ball bearing - schedule future maintenance
#  3 - segment is .021 defect of ball bearing - accelerate maintenance schedule
#  4 - segment is .028 defect of ball bearing - emergency maintenance
#
from awscrt import io, mqtt, auth, http
from awsiot import mqtt_connection_builder
import time as t
import json

import pandas as pd
import numpy as np
import datetime
import os.path
from IPython.core.pylabtools import figsize
from IPython.display import display
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras import Sequential
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.models import load_model


# this routine plots a segment of sensor data
def plot_segment(x,y,ttl, xlab, ylab):
    figsize(13,6)
    fig, ax = plt.subplots()
    plt.plot(x,y)
    plt.title(ttl, fontsize=16)
    plt.ylabel(ylab, fontsize=16)
    plt.xlabel(xlab, fontsize=16)
    plt.grid(True)
    for tick in ax.get_xticklabels():
        tick.set_fontsize(14)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(14)
    plt.show()
#
# Setup for connection to MQTT IoT Core
#
# Define ENDPOINT, CLIENT_ID, PATH_TO_CERT, PATH_TO_KEY, PATH_TO_ROOT, MESSAGE, TOPIC, and RANGE
ENDPOINT = "a36en74n1a2rs-ats.iot.us-east-1.amazonaws.com"
CLIENT_ID = "testDevice"
PATH_TO_CERT = "c:\\Thing\\certs\\d79f51ed34-certificate.pem.crt"
PATH_TO_KEY = "c:\\Thing\\certs\\d79f51ed34-private.pem.key"
PATH_TO_ROOT = "c:\\Thing\\certs\\root.pem"
TOPIC = "pjs/testing"
RANGE = 20

# Spin up resources
event_loop_group = io.EventLoopGroup(1)
host_resolver = io.DefaultHostResolver(event_loop_group)
client_bootstrap = io.ClientBootstrap(event_loop_group, host_resolver)
mqtt_connection = mqtt_connection_builder.mtls_from_path(
            endpoint=ENDPOINT,
            cert_filepath=PATH_TO_CERT,
            pri_key_filepath=PATH_TO_KEY,
            client_bootstrap=client_bootstrap,
            ca_filepath=PATH_TO_ROOT,
            client_id=CLIENT_ID,
            clean_session=False,
            keep_alive_secs=6
            )
print("Connecting to {} with client ID '{}'...".format(ENDPOINT, CLIENT_ID))
# Make the connect() call
connect_future = mqtt_connection.connect()
# Future.result() waits until a result is available
connect_future.result()
print("Connected!")
# Begin to publish only segments which show defective signals
print('Begin Publishing')

# 5 classes and each segment is 256 in size (number of samples)
segment_size = 256  # size of number of samples in a feature
num_classes = 5 # 0 through 4

# load model from a trained CNN
model = load_model('c:\\trainedmodel\\cnnmodel.h5')

# read the test set dataframe and convert to numpy arrays
df_test = pd.read_csv("C:\\trainedmodel\\testset.csv", index_col=0)
test_labels = np.array(df_test['labels'])
df_test = df_test.drop('labels', axis = 1)
test_features = df_test.to_numpy()
test_feature_length = len(test_features)

# reshape features to 3D for input to fit model
test_features = test_features.reshape(test_feature_length,segment_size,1)

# create categorical one hots for train and test labels
test_labels_one_hot = tf.keras.utils.to_categorical(test_labels, num_classes)

# process signal data as if the sensor was streaming it to the program
# CNN as input uses 256 byte segments. As soon as the stream produces
# 256 samples, the CNN model predicts the class of the signal (0 throught 4)
# The model was 100% accurate
print("Number of Segment Samples: ", test_feature_length)
for seg in range(4200,test_feature_length):
    tf = test_features[seg,:,0]
    tfcnn = tf.reshape(1,segment_size,1)
    predicted_class = model.predict(tfcnn)
    predicted_class = np.argmax(np.round(predicted_class),axis=1)[0]
    if predicted_class != test_labels[seg] :
        print("predicted class: ",predicted_class," actual class: ",test_labels[i]) 
    print(datetime.datetime.now(),seg, predicted_class)
    for sn in range(0,segment_size):
        if seg >= 4200 :
            message = {}
            message['device'] = 'accelerometer-motorend'
            message['datetimestamp'] = str(datetime.datetime.now())
            message['sample'] = str(tf[sn])
            message['samplepos'] = str(sn)
            message['segmentnum'] = str(seg)
            message['classnum'] = str(predicted_class)
            messageJson = json.dumps(message)
            mqtt_connection.publish(topic=TOPIC, payload=messageJson, qos=mqtt.QoS.AT_LEAST_ONCE)
            t.sleep(0.025)
print('Publish End')
disconnect_future = mqtt_connection.disconnect()
disconnect_future.result()








