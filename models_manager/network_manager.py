import re
import time
import copy
import os
import numpy as np
import tensorflow as tf
from collections import Counter
from tensorflow import keras
import pandas as pd
from pandas import Series
import multiprocessing
from pylab import rcParams
from sklearn.preprocessing import MinMaxScaler
from random import seed as random_seed
from random import random
from sklearn.model_selection import train_test_split
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters,MinimalFCParameters,EfficientFCParameters
from imblearn.under_sampling import NearMiss
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.metrics import classification_report
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as Kbackend
from tensorflow.keras.layers import Dropout, Activation, Dense, LSTM,Flatten,LayerNormalization,Multiply,Lambda
from tensorflow.python.keras.layers import CuDNNLSTM,Conv1D, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import Precision,Recall
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Embedding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.feature_selection import SelectPercentile,SelectKBest, chi2,f_classif
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
import settings
from support.participant_entry import get_participant_timestamp_entry_class,get_participant_entry_class
from support.connect_with_db import \
    batch_db_read,extract_sorted_collection
from models_manager.policies.WwsxParticipantsTracker_policy import WwsxParticipantsTrackerPolicy
from support.timeseries_features_entry import get_timeseries_feature_class,get_activity_features_timeseries_feature_class
from support.frailty_prediction_entry import get_frailty_prediction_entry
from contextlib import redirect_stdout
import tensorflow_addons as tfa
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import reset_default_graph

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
rcParams['figure.figsize'] = 14, 8

RANDOM_SEED = 0
ct=multiprocessing.get_context('spawn')
np.random.seed(RANDOM_SEED)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


config = ConfigProto()
config.gpu_options.allow_growth = True

ExtentedParameters = MinimalFCParameters()
ExtentedParameters['kurtosis'] = None
ExtentedParameters['abs_energy'] = None
ExtentedParameters['skewness'] = None
del ExtentedParameters['length']
log_path = "model_log_file.txt"  # it will be created automatically


def reset_seeds():
    np.random.seed(1)
    random_seed(2)
    if tf.__version__[0] == '2':
        tf.random.set_seed(3)
    else:
        tf.set_random_seed(3)
    print("RANDOM SEEDS RESET")

def model_builder(hp):
    model = Sequential()
    model.add(Flatten(input_shape=(77, )))

    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 32-512
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(Dense(units=hp_units, activation=hp.Choice("activation", ["relu", "tanh"])))
    model.add(Dense(3, activation="softmax"))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# Function to split a list into specified number of sublists
def split_list(lst, num_splits):
    avg = -(-len(lst) // num_splits)  # Ceiling division
    return [lst[i:i + avg] for i in range(0, len(lst), avg)]

class TransformerClassifier(tf.keras.Model):
    def __init__(self, num_classes, d_model, num_heads, dff, input_sequence_length, dropout_rate=0.1):
        super(TransformerClassifier, self).__init__()
        self.input_sequence_length=input_sequence_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_size = d_model // num_heads  # Calculate the size of each attention head

        # Input embedding layer
        self.embedding = Dense(d_model, input_shape=(input_sequence_length,))

        # Multi-head self-attention layer
        self.attention = tfa.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model,head_size=self.head_size)

        # Feed-forward layer
        self.feed_forward = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])

        # Layer normalization and dropout
        self.layer_norm1 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)

        self.layer_norm2 = LayerNormalization(epsilon=1e-6)
        self.dropout2 = Dropout(dropout_rate)

        # Output layer
        self.output_layer = Dense(num_classes, activation='softmax')

    def call(self, inputs, training=True):
        # Input embedding
        x = self.embedding(inputs)

        # Multi-head self-attention
        batch_size, input_sequence_length, d_model = tf.shape(x)[0], tf.shape(x)[1], self.d_model

        attention_output = self.attention(x, x,input_shape=(batch_size, input_sequence_length, d_model))
        x = self.dropout1(attention_output, training=training)
        x = self.layer_norm1(x + x)  # Residual connection

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.dropout2(ff_output, training=training)
        x = self.layer_norm2(x + x)  # Residual connection

        # Output classification
        x = self.output_layer(x)
        return x


class MyHyperModel(HyperModel):
    def __init__(self, num_classes,input_shape):
        self.num_classes = num_classes
        self.input_shape=input_shape

    def build(self, hp):
        input_shape=self.input_shape
        model = Sequential()
        model.add(keras.Input(shape=(input_shape,)))  # Replace input_shape with your input shape
        for i in range(2):  # Two layers
            units = hp.Int(f'units_{i}', min_value=32, max_value=512, step=32)
            activation = hp.Choice(f'activation_{i}', values=['relu', 'sigmoid', 'tanh'])
            model.add(tf.keras.layers.Dense(units=units, activation=activation))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(
            optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

class NetworkManager:
    def __init__(self,classification_classes=3,excluded_features=None):

        self.timestamp_str= 'timestamp'
        self.ecg_heart_rate_str = 'ecg_heart_rate'
        self.ecg_signal_quality_str = 'ecg_signal_quality'
        self.ecg_rr_interval_str = 'ecg_rr_interval'
        self.ecg_heart_rate_variability_str = 'ecg_heart_rate_variability'
        self.acceleration_x_str = 'acceleration_x'
        self.acceleration_y_str = 'acceleration_y'
        self.acceleration_z_str = 'acceleration_z'
        self.health_status_str = "health_status"
        self.participant_id_str='participant_id'
        self.feature_id_str='feature_id'
        self.breathing_rate_str= "breathing_rate"
        self.classification_classes=classification_classes
        self.excluded_features=excluded_features
        self.activity_class_list=[0,1,2,3,4,5]

        if self.classification_classes==3:
            self.frail_status_to_number = settings.frail_to_number_3_classes
        elif self.classification_classes==2:
            self.frail_status_to_number = settings.frail_to_number_2_classes

    def create_lstm_dataset(self,dataset, look_back=1):
        dataX, dataY = [], []

        seq_len=look_back+1

        for i in range(len(dataset) - seq_len):
            # takes
            a = dataset[i:(i + seq_len), 0]

            dataX.append(a)
            dataY.append(dataset[i + seq_len, 0])  #+1 in test mode
        return np.array(dataX), np.array(dataY)

    def inverse_window_x(self,data_x,window_segment_size=1):

        # Number of sublists to split each sublist inside k
        num_splits = window_segment_size

        # Apply the split_list function to each sublist in k
        split_k = [split_list(sublist, num_splits) for sublist in data_x]

        # Transpose the split sublists and flatten them
        result = [item for sublist in zip(*split_k) for item in sublist]

        return result

    def create_window_x(self,data_x,window_segment_size=1):

        new_data=[]
        for i in range(0, len(data_x), window_segment_size):
            window_data=[]
            for j in range(window_segment_size):
                window_data=window_data+data_x[i+j]
            new_data.append(window_data)
        return new_data

    def inverse_window_y(self,data_y,window_segment_size=1):

        num_elements_per_pair=window_segment_size
        expanded_labels = [label for label in data_y for _ in range(num_elements_per_pair)]

        return expanded_labels

    def create_window_y(self,data_y,window_segment_size=1):

        num_elements_per_pair = window_segment_size
        original_labels = [data_y[i] for i in range(0, len(data_y), num_elements_per_pair)]

        return original_labels

    def min_max_normalization(self,data_x,scaler=None,window_segment_size=1):
        from sklearn.preprocessing import MinMaxScaler

        if scaler is None:
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(np.array(data_x))
        else:
            normalized_data = scaler.transform(np.array(data_x))

        return normalized_data.tolist(),scaler


    def create_dataset_lstm(self,datasetX,datasetY, look_back=1):
        dataX, dataY = [], []

        seq_len=look_back+1

        resultDatasetSize =int(len(datasetX)/look_back)

        for i in range(resultDatasetSize):
            a = datasetX[i*look_back:(i+1)*look_back]
            #flattened_list = [item for sublist in a for item in sublist]
            flattened_list = [sublist for sublist in a]
            dataX.append(flattened_list)
            dataY.append(datasetY[(i+1)*look_back])

        return np.array(dataX), np.array(dataY)

    def extract_data_to_json(self):

        participant_entries = batch_db_read(get_participant_entry_class())

        participant_ids_dict = {'Frail':[],'Pre-frail':[],'Non-frail':[]}
        skip_multi_health_status_participants=True
        device_source='wwsx'
        if not skip_multi_health_status_participants:
            participant_ids_dict['Multi-Health-Status']=[]

        for participant_entry in participant_entries:
            participant_id = participant_entry.get_participant_id()

            participant_frailty_status_list = participant_entry.get_participant_frailty_status()

            if skip_multi_health_status_participants:
                if len(participant_frailty_status_list)>1:
                    continue

                health_status = participant_frailty_status_list[0]['frailty_status']
                participant_ids_dict[health_status].append(participant_id)
            else:
                if len(participant_frailty_status_list) > 1:
                    health_status='Multi-Health-Status'
                else:
                    health_status = participant_frailty_status_list[0]['frailty_status']
                participant_ids_dict[health_status].append(participant_id)

        if skip_multi_health_status_participants:
            data_length = [len(participant_ids_dict['Frail']),len(participant_ids_dict['Pre-frail']),len(participant_ids_dict['Non-frail'])]
            train_weights = self.calculate_train_weights(data_length)
        batch_limit=20

        participant_json={}
        health_status_index=0
        train_weights = [4,8,8]
        for health_status_id, participant_ids_list in participant_ids_dict.items():
            part_per_class = int(train_weights[-health_status_index])*self.classification_classes
            participant_data_index = 0

            health_status_index = health_status_index+1
            for participant_id in participant_ids_list:
                participant_timestamp_class = get_participant_timestamp_entry_class(device_source, participant_id)

                print('Participant ID', participant_id)

                if participant_id != '2109':
                    pass

                ind_timestamp = 0
                last_timestamp = None
                batch_limit=part_per_class
                participant_json[participant_id]={}
                while ind_timestamp < part_per_class:
                    participant_timestamp_entries = extract_sorted_collection(participant_timestamp_class,
                                                                              limit=batch_limit,
                                                                              low_timestamp_threshold=last_timestamp)
                    for participant_timestamp_entry in participant_timestamp_entries:
                        participant_json[participant_id][participant_timestamp_entry.get_timestamp()]=participant_timestamp_entry.__dict__
                        ecg_heart_rate = participant_timestamp_entry.get_ecg_heart_rate()
                        ecg_signal_quality = participant_timestamp_entry.get_ecg_signal_quality()
                        ecg_rr_interval = participant_timestamp_entry.get_ecg_rr_interval()
                        ecg_heart_rate_variability = participant_timestamp_entry.get_ecg_heart_rate_variability()
                        acceleration_x = participant_timestamp_entry.get_acceleration_x()
                        acceleration_y = participant_timestamp_entry.get_acceleration_y()
                        acceleration_z = participant_timestamp_entry.get_acceleration_z()
                        health_status = participant_timestamp_entry.get_health_status()
                        breathing_rate = participant_timestamp_entry.get_breathing_rate()
                        participant_json[participant_id][participant_timestamp_entry.get_timestamp()]={}
                        participant_json[participant_id][participant_timestamp_entry.get_timestamp()]['ecg_heart_rate']=ecg_heart_rate
                        participant_json[participant_id][participant_timestamp_entry.get_timestamp()]['ecg_signal_quality'] = ecg_signal_quality
                        participant_json[participant_id][participant_timestamp_entry.get_timestamp()][
                            'ecg_rr_interval'] = ecg_rr_interval
                        participant_json[participant_id][participant_timestamp_entry.get_timestamp()][
                            'ecg_heart_rate_variability'] = ecg_heart_rate_variability
                        participant_json[participant_id][participant_timestamp_entry.get_timestamp()][
                            'acceleration_x'] = acceleration_x
                        participant_json[participant_id][participant_timestamp_entry.get_timestamp()][
                            'acceleration_y'] = acceleration_y
                        participant_json[participant_id][participant_timestamp_entry.get_timestamp()][
                            'acceleration_z'] = acceleration_z
                        participant_json[participant_id][participant_timestamp_entry.get_timestamp()][
                            'health_status'] = health_status
                        participant_json[participant_id][participant_timestamp_entry.get_timestamp()][
                            'breathing_rate'] = breathing_rate
                    ind_timestamp=ind_timestamp+1

                participant_data_index = participant_data_index+1
                if participant_data_index==5:
                    print(health_status_id)
                    break
            print(participant_json)
            import json
            with open("data.json", "w") as json_file:
                json.dump(participant_json, json_file)

    def build_window_segment(self,features_per_participant,
                                  class_per_participant,
                                  window_segments_size,
                                  predict_per_participant_health_status=False):

        participant_entries_X=[]
        participant_entries_Y=[]
        features_per_participant_new={}
        class_per_participant_new={}
        windows_list = []
        window_with_change=0

        for participant_id, entries_X in features_per_participant.items():
            index_in_participant = 0
            index_in_window = 0
            part_classes = class_per_participant[participant_id]
            last_class = part_classes[index_in_window]
            new_class=last_class
            window_segment = []
            features_per_participant_new[participant_id]=[]
            class_per_participant_new[participant_id]=[]

            for entryX in entries_X:

                last_class=new_class
                new_class = part_classes[index_in_participant]

                if new_class != last_class:
                    window_with_change=window_with_change+1
                    if index_in_window>1:
                        window_segment = []
                        index_in_window = 0


                if index_in_window==(window_segments_size):
                    #close window open new

                    participant_entries_X.append(window_segment)
                    participant_entries_Y.append(new_class)
                    features_per_participant_new[participant_id].append(window_segment)
                    class_per_participant_new[participant_id].append(new_class)
                    window_segment = []
                    index_in_window = 0

                window_segment=window_segment+entryX

                index_in_participant = index_in_participant + 1
                index_in_window=index_in_window+1
        print('WINDOW WITH CHANGE',window_with_change)

        return participant_entries_X,participant_entries_Y,features_per_participant_new,class_per_participant_new

    def balanced_train_test_split(self,X, y, test_size=0.2, random_state=None):

        #stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=random_state, stratify=y)


        return X_train, X_test, y_train, y_test

    def get_feature_windows(self,timeseries_feature_entry,aggregated_dataset):
        if aggregated_dataset:
            aggregated_features = timeseries_feature_entry.get_aggregated_features(self.activity_class_list)

            feature_windows = aggregated_features
        else:
            ecg_heart_rate_features = timeseries_feature_entry.get_ecg_heart_rate_features()
            ecg_signal_quality_features = timeseries_feature_entry.get_ecg_signal_quality_features()
            ecg_rr_interval_features = timeseries_feature_entry.get_ecg_rr_interval_features()
            acceleration_x_features = timeseries_feature_entry.get_acceleration_x_features()
            acceleration_y_features = timeseries_feature_entry.get_acceleration_y_features()
            acceleration_z_features = timeseries_feature_entry.get_acceleration_z_features()
            breathing_rate_features = timeseries_feature_entry.get_breathing_rate_features()
            feature_windows = (ecg_heart_rate_features + ecg_signal_quality_features +
                               ecg_rr_interval_features + acceleration_x_features+
                               acceleration_y_features+acceleration_z_features+
                               breathing_rate_features)
        return feature_windows

    def train_model_via_features(self,
                                 model='ann',
                                 data_processing_algorith='smote',
                                 feature_selection_method=None,
                                 feature_selection_size=50,
                                 features_window_size_id = '24hr',
                                 features_size= 'basic',
                                 window_segments_size=1,
                                 epoch_times=None,
                                 data_split_strategy='time',
                                 aggregated_dataset=False):
        """
        :param model: valid valued = svm,ann,knn,gradientboosting
        :param data_processing_algorith: valid valued = smote,tomeklinks,weights,hybrid
        :param features_window_size_id: 2m,10m,5000,24hr
        :return: accuracy
        """

        features_list=MinimalFCParameters().keys()
        if features_size.capitalize()=='Extended' or features_size.capitalize()=='Full':
            features_list=ExtentedParameters.keys()

        features_list = list(features_list)
        if self.excluded_features is not None:
            for excluded_feature in self.excluded_features:
                if excluded_feature in features_list:
                    features_list.remove(excluded_feature)

        if aggregated_dataset:
            get_timeseries_feature_class_to_use=get_activity_features_timeseries_feature_class
        else:
            get_timeseries_feature_class_to_use=get_timeseries_feature_class

        features_number = len(features_list)
        inputSize = features_number*7*window_segments_size
        model=model.lower()

        smote = SMOTE()

        participant_ids_dict = {'Frail': [], 'Pre-frail': [], 'Non-frail': []}
        features_per_participant={}
        class_per_participant={}
        participant_entries_X = []
        participant_entries_Y = []

        batch_limit = 50000
        participant_id = None
        ind_timestamp = batch_limit + 1

        while ind_timestamp >= batch_limit:
            print('NEW BATCH-', participant_id)
            timeseries_feature_entries = extract_sorted_collection(get_timeseries_feature_class_to_use(features_window_size_id,features_size),
                                                                      limit=batch_limit,
                                                                      low_timestamp_threshold=participant_id,
                                                                      timestamp_variable='participant_id')
            ind_timestamp = 0
            participant_classes=[]
            for timeseries_feature_entry in timeseries_feature_entries:

                health_status = timeseries_feature_entry.get_health_status()
                participant_id = timeseries_feature_entry.get_participant_id()
                participant_ids_dict[health_status].append(timeseries_feature_entry)
                timeseries_feature_entry.init_features(features_list)
                entries_X =self.get_feature_windows(timeseries_feature_entry,aggregated_dataset)
                health_status_class = self.frail_status_to_number[health_status]

                if (data_split_strategy.lower()=='time' or window_segments_size>1 or
                        data_split_strategy.lower()=='participant' or data_split_strategy.lower()=='hybrid'):
                    if participant_id not in features_per_participant.keys():
                        features_per_participant[participant_id] = []
                        class_per_participant[participant_id]=[]
                        participant_classes.append(health_status_class)
                    features_per_participant[participant_id].append(entries_X)
                    class_per_participant[participant_id].append(health_status_class)
                else:
                    #predict per time series
                    participant_entries_X.append(entries_X)
                    participant_entries_Y.append(health_status_class)

                ind_timestamp = ind_timestamp + 1

        if window_segments_size > 1:

            (participant_entries_X,
             participant_entries_Y,
             features_per_participant_new,
             class_per_participant_new) = self.build_window_segment(features_per_participant,
                                                                    class_per_participant,
                                                                    window_segments_size)

            features_per_participant = features_per_participant_new.copy()
            class_per_participant = class_per_participant_new.copy()

        if data_split_strategy.lower()=='participant':

            # Convert the dictionary into lists of participants, data, and labels
            participants = list(features_per_participant.keys())
            data_lists = list(features_per_participant.values())
            labels_lists = [class_per_participant[participant] for participant in participants]
            prevailed_class = []

            for participant_id, classes in class_per_participant.items():
                class_counters= [class_per_participant[participant_id].count(0),
                                 class_per_participant[participant_id].count(1),
                                 class_per_participant[participant_id].count(2)]
                max_class = max(class_counters)
                max_index = class_counters.index(max_class)
                max_class = max_index
                prevailed_class.append(max_class)
            # Split the data and labels into train and test sets
            train_participants, test_participants, train_data, test_data, train_labels, test_labels = train_test_split(
                participants, data_lists, labels_lists, test_size=0.1, stratify=prevailed_class
            )

            x_train=[]
            for participant_entries_list in train_data:
                x_train= x_train + participant_entries_list

            x_test=[]
            for participant_entries_list in test_data:
                x_test = x_test + participant_entries_list

            y_train=[]
            y_test=[]

            for train_label_seq in train_labels:
                for train_label in train_label_seq:
                    y_train.append(train_label)

            for test_label_seq in test_labels:
                for test_label in test_label_seq:
                    y_test.append(test_label)

        elif data_split_strategy.lower()=='time':
            x_train=[]
            y_train=[]
            x_test=[]
            y_test=[]
            total_same_status=0
            total_diff_status=0
            for participant_id,entries_X in features_per_participant.items():
                if len(entries_X)<3:
                    continue
                participant_x_train, participant_x_test, participant_y_train, participant_y_test = train_test_split(entries_X,
                                                                                                                    class_per_participant[participant_id],
                                                                                                                    test_size=0.4,
                                                                                                                    shuffle=False)
                part_index=0
                for participant in participant_x_train:
                    x_train.append(participant)
                    y_train.append(participant_y_train[part_index])
                    last_status_of_participant = participant_y_train[part_index]
                    part_index=part_index+1

                part_index = 0

                for participant in participant_x_test:
                    x_test.append(participant)
                    y_test.append(participant_y_test[part_index])
                    if participant_y_test[part_index]!=last_status_of_participant:
                        total_diff_status=total_diff_status+1
                    else:
                        total_same_status=total_same_status+1
                    part_index = part_index + 1

        elif data_split_strategy.lower() == 'hybrid':
            participants = list(features_per_participant.keys())
            data_lists = list(features_per_participant.values())
            labels_lists = [class_per_participant[participant] for participant in participants]
            prevailed_class = []

            for participant_id, classes in class_per_participant.items():
                class_counters = [class_per_participant[participant_id].count(0),
                                  class_per_participant[participant_id].count(1),
                                  class_per_participant[participant_id].count(2)]
                max_class = max(class_counters)
                max_index = class_counters.index(max_class)
                max_class=max_index
                if class_per_participant[participant_id].count(2)>0:
                    max_class=2
                prevailed_class.append(max_class)

            # Split the data and labels into train and test sets
            train_participants, test_participants, train_data, test_data, train_labels, test_labels = train_test_split(
                participants, data_lists, labels_lists, test_size=0.3, stratify=prevailed_class
            )

            x_train = []
            y_train = []
            y_test = []
            x_test = []
            for participant_entries_list in train_data:
                x_train = x_train + participant_entries_list

            for train_label_seq in train_labels:
                for train_label in train_label_seq:
                    y_train.append(train_label)

            test_part_index=0
            for participant_id in test_participants:
                entries_X = test_data[test_part_index]
                entries_y= test_labels[test_part_index]
                if len(entries_y)>10:
                    group_b_x_train, group_b_x_test, group_b_y_train, group_b_y_test = train_test_split(entries_X,
                                                                                                        entries_y,
                                                                                                        test_size=0.9,
                                                                                                        shuffle=False)
                elif len(entries_y)>1:
                    group_b_x_train, group_b_x_test, group_b_y_train, group_b_y_test = train_test_split(entries_X,
                                                                                                        entries_y,
                                                                                                        test_size=0.5,
                                                                                                        shuffle=False)
                else:
                    continue


                x_train = x_train + group_b_x_train
                x_test = x_test + group_b_x_test
                y_train=y_train+group_b_y_train
                y_test=y_test+group_b_y_test

                test_part_index=test_part_index+1


        else:

            x_train, x_test, y_train, y_test = self.balanced_train_test_split(np.array(participant_entries_X), np.array(participant_entries_Y), test_size=0.2)
            x_train=x_train.tolist()
            x_test = x_test.tolist()
            y_train=y_train.tolist()
            y_test=y_test.tolist()

        if window_segments_size>1:
            x_train = self.inverse_window_x(x_train,window_segments_size)
            x_train = np.array(x_train)
            

            x_test = self.inverse_window_x(x_test,window_segments_size)
            x_test = np.array(x_test)

            y_train = self.inverse_window_y(y_train,window_segments_size)
            print('INVERSE',len(x_train), len(y_train))


        X_train_preprocessed, scaler = self.min_max_normalization(x_train,window_segment_size=window_segments_size)

        X_test_preprocessed, scaler = self.min_max_normalization(x_test,scaler,window_segment_size=window_segments_size)

        feature_names=[]
        attr_list=['ecg_heart_rate',
                   'ecg_signal_quality',
                   'ecg_rr_interval',
                   'acceleration_x',
                   'acceleration_y',
                   'acceleration_z',
                   'breathing_rate']

        for attr in attr_list:
            for feature in features_list:
                feature_names.append(attr+'__'+feature)

        print(y_train.count(0))
        print(y_train.count(1))
        print(y_train.count(2))

        selected_feature_names = feature_names
        if feature_selection_method is not None:
            print('FEATURE SELECTION',feature_selection_method)
            if feature_selection_method=='selectkbest':

                k_best = feature_selection_size
                selector = SelectKBest(score_func=chi2, k=k_best)
                X_train_preprocessed = selector.fit_transform(X_train_preprocessed, y_train)
                X_train_preprocessed=X_train_preprocessed.tolist()

                # Get the mask of selected features
                selected_mask = selector.get_support()

                # Print the names of the selected features
                selected_feature_names = selected_mask
                X_test_preprocessed = np.array(X_test_preprocessed)
                X_test_preprocessed = X_test_preprocessed[:, selected_mask]
                X_test_preprocessed = X_test_preprocessed.tolist()

                print("Selected Features")

                inputSize=k_best
            elif feature_selection_method=='selectpercentile':
                percentile = feature_selection_size
                selector = SelectPercentile(score_func=chi2, percentile=percentile)
                X_train_preprocessed = selector.fit_transform(X_train_preprocessed, y_train)
                X_train_preprocessed = X_train_preprocessed.tolist()

                X_test_preprocessed = selector.transform(X_test_preprocessed)
                X_test_preprocessed = X_test_preprocessed.tolist()

                inputSize=int(inputSize*feature_selection_size/100)

        if window_segments_size>1:
            X_train_preprocessed = self.create_window_x(X_train_preprocessed,window_segments_size)
            X_test_preprocessed = self.create_window_x(X_test_preprocessed,window_segments_size)

            #print(x_train_normalized)

            y_train = self.create_window_y(y_train,window_segments_size)

            print( len(y_train),len(X_train_preprocessed))

        if data_processing_algorith is None:
            # No preprocessing
            pass
        elif data_processing_algorith.lower()=='smote':
            X_train_preprocessed, y_train_preprocessed = smote.fit_resample(X_train_preprocessed, y_train)
        elif data_processing_algorith.lower()=='nearmiss':
            nm = NearMiss()
            X_train_preprocessed, y_train_preprocessed = nm.fit_resample(X_train_preprocessed, y_train)
        elif data_processing_algorith=='hybrid':
            # Create separate SMOTE and TomekLinks objects
            smote = SMOTE()
            tomek = TomekLinks()

            # First, apply SMOTE to oversample the minority class
            X_oversampled, y_oversampled = smote.fit_resample(X_train_preprocessed, y_train)

            # Then, apply TomekLinks to remove the Tomek links (instances that are considered noisy)
            X_train_preprocessed, y_train_preprocessed = tomek.fit_resample(X_oversampled, y_oversampled)
        elif data_processing_algorith=='tomeklinks':
            tomek = TomekLinks()
            X_train_preprocessed, y_train_preprocessed = tomek.fit_resample(X_train_preprocessed, y_train)
        else:
            # default
            pass

        if data_processing_algorith is not None:
            print('DATA PREPROCESSING:',data_processing_algorith)
        print('MODEL:', model)

        class_weights=None
        if data_processing_algorith is not None:
            if data_processing_algorith.lower() == 'weights':
                from sklearn.utils import class_weight
                class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                                  classes = np.unique(y_train),
                                                                  y = y_train)
                print(class_weights)
                class_weights = dict(enumerate(class_weights))
                y_train_preprocessed=y_train

        x_train_normalized = X_train_preprocessed
        x_test_normalized = X_test_preprocessed
        y_train=y_train_preprocessed
        patienceModelStop = 100
        f1=1
        precision=1
        recall=1

        if aggregated_dataset:
            inputSize=840

        print('-----TRAIN DATA-----')
        print(len(x_train_normalized))
        print(y_train.count(0))
        print(y_train.count(1))
        print(y_train.count(2))
        print('-----TEST DATA-----')
        print(len(x_test_normalized))
        print(y_test.count(0))
        print(y_test.count(1))
        print(y_test.count(2))


        if model=='svm':
            #from sklearn.svm import SVC
            from thundersvm import SVC

            from sklearn.metrics import accuracy_score
            #from thundersvmsklearn.svm import SVC as cudaSVC


            #svm_model = SVC(kernel='linear',
            #                class_weight='balanced')

            #svm_model = SVC(kernel='rbf', C=0.9, decision_function_shape='ovr',class_weight=class_weights)  # OvR for multi-class
            x_train_normalized = np.array(x_train_normalized)
            y_train = np.array(y_train)
            x_test_normalized = np.array(x_test_normalized)
            y_test = np.array(y_test)

            svm_model = SVC(kernel='linear', C=1.0, decision_function_shape='ovo')
            print('SVM START')
            svm_model.fit(x_train_normalized, y_train)
            print('SVM TRAIN COMPLETED')
            y_pred = svm_model.predict(x_test_normalized)

            f1 = f1_score(y_test, y_pred, average='macro')
            precision = precision_score(y_test, y_pred, average='macro')
            # Evaluate the classifier
            accuracy = accuracy_score(y_test, y_pred)
            score = [None, accuracy]
            recall = recall_score(y_test, y_pred, average='macro')

            print("Test accuracy:", accuracy)
            print("F1 Score:", f1)
            print("Precision:", precision)
            print("Recall:", recall)

        elif model.lower()=='ann':
            print('-----ARTIFICIAL NEURAL NETWORK-----')
            print(len(x_train_normalized))
            print(len(y_train))

            print('---')
            print(len(x_test_normalized))
            print(len(y_test))

            x_train_normalized = np.array(x_train_normalized)
            y_train = np.array(y_train)

            # Build the feedforward neural network model
            model = Sequential()
            model.add(Dense(256, activation='relu', input_shape=(inputSize,)))
            model.add(Dropout(0.1))
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.1))
            model.add(Dense(self.classification_classes, activation='softmax'))
            model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            early_stopping = EarlyStopping(monitor='val_accuracy', patience=patienceModelStop, restore_best_weights=True)

            if epoch_times is None:
                epoch_times = 1000
                model.fit(x_train_normalized, y_train, epochs=epoch_times,class_weight=class_weights,
                          validation_data=(x_test_normalized,y_test),callbacks=[early_stopping])
                number_of_epochs_stopped = early_stopping.stopped_epoch
                if number_of_epochs_stopped != 0:
                    epoch_times = number_of_epochs_stopped-patienceModelStop

                print("Number of epochs at which early stopping was triggered:", epoch_times)

                if epoch_times<200:
                    epoch_times=200
            else:
                model.fit(x_train_normalized, y_train, epochs=epoch_times, class_weight=class_weights)

            score = model.evaluate(x_test_normalized, y_test)

            y_pred = model.predict(x_test_normalized, batch_size=64, verbose=1)
            y_pred_bool = np.argmax(y_pred, axis=1)
            f1 = f1_score(y_test, y_pred_bool, average='macro')
            print("F1 Score:", f1)
            precision = precision_score(y_test, y_pred_bool, average='macro')
            print("Precision:", precision)
            recall = recall_score(y_test, y_pred_bool,average='macro')
            print("Recall:", recall)
            print(classification_report(y_test, y_pred_bool))

        elif model.capitalize() == 'Cnn':
            print('-----CONVOLUTION NEURAL NETWORK-----')
            print(len(x_train_normalized))
            print(len(y_train))

            print('---')
            print(len(x_test_normalized))
            print(len(y_test))

            x_train_normalized = np.array(x_train_normalized)
            y_train = np.array(y_train)
            # Build the 1D CNN model
            model = Sequential()
            model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(inputSize,1)))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(self.classification_classes, activation='softmax'))

            model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            early_stopping = EarlyStopping(monitor='val_accuracy', patience=patienceModelStop,
                                           restore_best_weights=True)

            x_train_normalized=np.array(x_train_normalized)
            x_test_normalized=np.array(x_test_normalized)
            y_train=np.array(y_train)
            y_test=np.array(y_test)
            x_train_normalized = x_train_normalized.reshape(x_train_normalized.shape[0], x_train_normalized.shape[1], 1)
            x_test_normalized = x_test_normalized.reshape(x_test_normalized.shape[0], x_test_normalized.shape[1], 1)

            if epoch_times is None:
                epoch_times = 1000
                model.fit(x_train_normalized, y_train, epochs=epoch_times, class_weight=class_weights,
                          validation_data=(x_test_normalized, y_test), callbacks=[early_stopping])
                number_of_epochs_stopped = early_stopping.stopped_epoch
                if number_of_epochs_stopped != 0:
                    epoch_times = number_of_epochs_stopped - patienceModelStop

                print("Number of epochs at which early stopping was triggered:", epoch_times)

                if epoch_times < 200:
                    epoch_times = 200
            else:
                model.fit(x_train_normalized, y_train, epochs=epoch_times, class_weight=class_weights)

            score = model.evaluate(x_test_normalized, y_test)

            y_pred = model.predict(x_test_normalized, batch_size=32, verbose=1)
            y_pred_bool = np.argmax(y_pred, axis=1)
            f1 = f1_score(y_test, y_pred_bool, average='macro')
            print("F1 Score:", f1)
            precision = precision_score(y_test, y_pred_bool, average='macro')
            print("Precision:", precision)
            recall = recall_score(y_test, y_pred_bool, average='macro')
            print("Recall:", recall)
            print(classification_report(y_test, y_pred_bool))

        elif model.capitalize()=='Decisiontree':
            from imblearn.ensemble import BalancedBaggingClassifier
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.metrics import accuracy_score
            # Create an instance
            classifier = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                                   sampling_strategy='not majority',
                                                   replacement=False,
                                                   random_state=42)
            classifier.fit(x_train_normalized, y_train)
            preds = classifier.predict(x_test_normalized)
            print(preds)
            print("Train data accuracy:", accuracy_score(y_true=y_train, y_pred=classifier.predict(x_train_normalized)))
            print("Test data accuracy:", accuracy_score(y_true=y_test, y_pred=preds))

        elif model.capitalize()=='Lstm':
            print('-----LONG-SHORT TERM NEURAL NETWORK-----')
            print(len(x_train_normalized))
            print(len(y_train))

            print('---')
            print(len(x_test_normalized))
            print(len(y_test))
            num_time_steps=4

            x_train_normalized = np.array(x_train_normalized)
            y_train = np.array(y_train)

            # Build the LSTM model
            model = Sequential()
            model.add(LSTM(128, input_shape=(num_time_steps, inputSize)))
            model.add(Dropout(0.15))
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.15))
            model.add(Dense(self.classification_classes, activation='softmax'))
            model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            early_stopping = EarlyStopping(monitor='val_accuracy', patience=patienceModelStop,
                                           restore_best_weights=True)

            x_train_normalized = np.array(x_train_normalized)
            x_test_normalized = np.array(x_test_normalized)
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            # Reshape input data for LSTM
            #x_train_normalized = x_train_normalized.reshape(x_train_normalized.shape[0], 1, inputSize)
            #x_test_normalized = x_test_normalized.reshape(x_test_normalized.shape[0], 1, inputSize)

            # Reshape the data
            x_train_normalized = np.array([x_train_normalized[i:i + num_time_steps] for i in range(len(x_train_normalized) - num_time_steps + 1)])

            # Reshape the data
            x_test_normalized = np.array(
                [x_test_normalized[i:i + num_time_steps] for i in range(len(x_test_normalized) - num_time_steps + 1)])

            # Since we are considering timesteps of 2, we remove the first label
            y_train = y_train[num_time_steps - 1:]
            y_test = y_test[num_time_steps - 1:]

            if epoch_times is None:
                epoch_times = 1000
                model.fit(x_train_normalized, y_train, epochs=epoch_times, class_weight=class_weights,
                          validation_data=(x_test_normalized, y_test), callbacks=[early_stopping])
                number_of_epochs_stopped = early_stopping.stopped_epoch
                if number_of_epochs_stopped != 0:
                    epoch_times = number_of_epochs_stopped - patienceModelStop

                print("Number of epochs at which early stopping was triggered:", epoch_times)

                if epoch_times < 200:
                    epoch_times = 200
            else:
                model.fit(x_train_normalized, y_train, epochs=epoch_times, class_weight=class_weights)

            score = model.evaluate(x_test_normalized, y_test)

            y_pred = model.predict(x_test_normalized, batch_size=32, verbose=1)
            y_pred_bool = np.argmax(y_pred, axis=1)
            f1 = f1_score(y_test, y_pred_bool, average='macro')
            print("F1 Score:", f1)
            precision = precision_score(y_test, y_pred_bool, average='macro')
            print("Precision:", precision)
            recall = recall_score(y_test, y_pred_bool, average='macro')
            print("Recall:", recall)
            print(classification_report(y_test, y_pred_bool))


        elif model.lower()=='gradientboosting':
            # Create the Gradient Boosting Classifier
            from sklearn.metrics import accuracy_score
            import xgboost as xgb

            if data_processing_algorith=='weights':
                model_classifier = xgb.XGBClassifier(objective='multi:softmax',
                                                     num_class=self.classification_classes,
                                                     sample_weight=class_weights,
                                                     n_estimators=300,  # Increase the number of trees
                                                     max_depth=6,  # Increase the tree depth
                                                     min_child_weight=3,  # Increase the minimum child weight
                                                     colsample_bytree=0.8,  # Use more features for each tree
                                                     reg_alpha=0.01,  # Apply L1 regularization
                                                     reg_lambda=1.0,  # Apply L2 regularization
                                                     random_state=42
                                                     )
            else:
                #model_classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=3)
                model_classifier = xgb.XGBClassifier(
                    objective='multi:softmax',
                    num_class=self.classification_classes,
                    n_estimators=300,  # Increase the number of trees
                    max_depth=6,  # Increase the tree depth
                    min_child_weight=3,  # Increase the minimum child weight
                    colsample_bytree=0.8,  # Use more features for each tree
                    reg_alpha=0.01,  # Apply L1 regularization
                    reg_lambda=1.0,  # Apply L2 regularization
                    random_state=42
                )

            # Train the classifier on the training data

            #y_train = to_categorical(y_train, num_classes=self.classification_classes)
            model_classifier.fit(x_train_normalized, y_train)

            # Make predictions on the test data
            y_pred = model_classifier.predict(x_test_normalized)

            f1 = f1_score(y_test, y_pred, average='macro')
            precision = precision_score(y_test, y_pred, average='macro')
            # Evaluate the classifier
            accuracy = accuracy_score(y_test, y_pred)
            score = [None, accuracy]
            recall = recall_score(y_test, y_pred, average='macro')

            print("Test accuracy:", accuracy)
            print("F1 Score:", f1)
            print("Precision:", precision)
            print("Recall:", recall)
            print(classification_report(y_test, y_pred))

        elif model.lower()=='knn':
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.metrics import accuracy_score

            # Create and train the KNN classifier
            k = 5  # You can adjust the value of k (number of neighbors) based on your preference
            knn_classifier = KNeighborsClassifier(n_neighbors=k)
            knn_classifier.fit(x_train_normalized, y_train)

            # Make predictions on the test set
            y_pred = knn_classifier.predict(x_test_normalized)

                #print(y_test)
                #print(y_pred)

            f1 = f1_score(y_test, y_pred, average='macro')
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred,average='macro')
            # Evaluate the classifier
            accuracy = accuracy_score(y_test, y_pred)
            score=[None, accuracy]
            print("Test accuracy:", accuracy)
            print("F1 Score:", f1)
            print("Precision:", precision)
            print("Recall:", recall)

        elif model.lower()=='autotune':
            num_classes = self.classification_classes
            input_shape = inputSize
            hypermodel = MyHyperModel(num_classes=num_classes,input_shape=input_shape)
            tuner = RandomSearch(
                hypermodel,
                objective='val_accuracy',  # Use 'val_loss' or other metrics as per your preference
                max_trials=30,  # Number of hyperparameter combinations to try
                directory='my_tuner_dirfinal',  # Directory to store the results of the tuning process
                project_name='my_neural_network')  # Name of the tuning project

            max_epoch = 800
            # Early stopping callback
            early_stopping = EarlyStopping(monitor='val_accuracy', patience=patienceModelStop, restore_best_weights=True)
            tuner.search(x_train_normalized, y_train, epochs=max_epoch,verbose=1,validation_data=(x_test_normalized, y_test), callbacks=[early_stopping])

            number_of_epochs_stopped = early_stopping.stopped_epoch
            if number_of_epochs_stopped==0:
                number_of_epochs_stopped=max_epoch

            print("Number of epochs at which early stopping was triggered:", number_of_epochs_stopped)
            best_epoch=max_epoch
            # Get the best hyperparameters and epoch
            best_model = tuner.get_best_models(1)[0]
            best_hyperparameters = tuner.get_best_hyperparameters(1)[0]

            best_model.fit(x_train_normalized, y_train, epochs=best_epoch,verbose=1,validation_data=(x_test_normalized, y_test), callbacks=[early_stopping])

            score = best_model.evaluate(x_test_normalized, y_test, verbose=0)
            accuracy=score[1]
            print("Accuracy: %.2f%%" % (score[1] * 100))

        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        return score[1],f1,precision,recall,epoch_times


    def evaluate_models_with_features(self,model='svm',
                                     data_processing_algorith='smote',
                                     feature_selection_method=None,
                                     feature_selection_size=40,
                                     features_size='Basic',
                                     features_window_size_id='24hr',
                                     window_segments_size=1,
                                     num_of_tests=3,
                                     epoch_times=None,
                                     data_split_strategy = 'time',
                                     aggregated_dataset=False):

        #SVM - SMOKE EVALUATION
        sum_accuracy=0
        sum_f1 = 0
        sum_precision=0
        sum_recall=0
        number_of_tests = num_of_tests
        accuracy_list=[]
        for i in range(number_of_tests):
            accuracy,f1,precision,recall,epoch_times = self.train_model_via_features(model=model,
                                                     data_processing_algorith=data_processing_algorith,
                                                     feature_selection_method=feature_selection_method,
                                                     feature_selection_size=feature_selection_size,
                                                     features_size=features_size,
                                                     window_segments_size=window_segments_size,
                                                     features_window_size_id=features_window_size_id,
                                                     epoch_times=epoch_times,
                                                     data_split_strategy=data_split_strategy,
                                                     aggregated_dataset=aggregated_dataset)
            accuracy_list.append(accuracy)
            sum_accuracy=sum_accuracy+accuracy
            sum_f1=sum_f1+f1
            sum_precision=sum_precision+precision
            sum_recall=sum_recall+recall
        average_accuracy = sum_accuracy/number_of_tests
        average_f1 = sum_f1/number_of_tests
        average_precision =sum_precision/number_of_tests
        average_recall = sum_recall / number_of_tests
        print('AVERAGE ACCURACY',average_accuracy)
        print('MAX',max(accuracy_list))
        print('MIN', min(accuracy_list))
        print('AVERAGE F1', average_f1)
        print('AVERAGE Precission', average_precision)
        with open(log_path, "a") as writefile:
            with redirect_stdout(writefile):
                print('MODEL:',model)
                print('DATA PREPROCESSING ALGORITHM:',data_processing_algorith)
                print('FEATURE SIZE:', features_size)
                print('FEATURE WINDOW ID:', features_window_size_id)
                if feature_selection_method is not None:
                    print('FEATURE SELECTION:',feature_selection_method)
                    print('FEATURES:', feature_selection_size)
                print('CLASSES:', self.classification_classes)
                print('AVERAGE ACCURACY: ',round(average_accuracy,5)*100)
                print('MAX ACCURACY: ', round(max(accuracy_list),5)*100)
                print('MIN ACCURACY: ', round(min(accuracy_list),5)*100)
                print('AVERAGE F1: ',round(average_f1,5)*100)
                print('AVERAGE PRECISION: ', round(average_precision,5)*100)
                print('AVERAGE RECALL: ', round(average_recall, 5)*100)
                writefile.write("\n")
                writefile.write("\n")

    def create_dataset(self,dataset, look_back=1):
        dataX, dataY = [], []

        seq_len=look_back+1

        for i in range(len(dataset) - seq_len):
            # takes
            a = dataset[i:(i + seq_len), 0]
            dataX.append(a)
            dataY.append(dataset[i + seq_len, 0])  #+1 in test mode
        return np.array(dataX), np.array(dataY)


def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

async def async_main():

    s = time.perf_counter()

    exluded_features = ['length']

    model_list = [ 'gradientboosting', 'knn','cnn', 'ann']
    data_processing_algorith = ['smote', 'hybrid', 'tomeklinks', 'weights']
    features_window_size_id_lst = ['7d']

    plugin = NetworkManager(3, exluded_features)

    for model_selected in model_list:
        for data_processing_algorith_selected in data_processing_algorith:
            for features_window_size_id_selected in features_window_size_id_lst:
                plugin.evaluate_models_with_features(model=model_selected,
                                                     data_processing_algorith=data_processing_algorith_selected,
                                                     features_size='full',
                                                     features_window_size_id=features_window_size_id_selected,
                                                     num_of_tests=3,
                                                     window_segments_size=1,
                                                     data_split_strategy='None',
                                                     feature_selection_method='selectpercentile',
                                                     aggregated_dataset=False,
                                                     feature_selection_size=95)
                plugin.evaluate_models_with_features(model=model_selected,
                                                     data_processing_algorith=data_processing_algorith_selected,
                                                     features_size='full',
                                                     features_window_size_id=features_window_size_id_selected,
                                                     num_of_tests=3,
                                                     window_segments_size=1,
                                                     data_split_strategy='None', aggregated_dataset=False,
                                                     feature_selection_method=None)


    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.2f} seconds.")

def main():
    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_until_complete(async_main())
