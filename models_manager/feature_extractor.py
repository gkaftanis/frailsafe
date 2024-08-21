from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters,MinimalFCParameters,EfficientFCParameters
import numpy as np
import time
import re
import pandas as pd
import settings
from support.participant_entry import get_participant_timestamp_entry_class,get_participant_entry_class
from support.connect_with_db import \
    batch_db_read,\
    receive_last_timestamp,\
    extract_sorted_collection,\
    count_collection_entries_with_timestamp,read_collection_keys,receive_participants_with_one_status
from support.timeseries_features_entry import get_timeseries_feature_class,get_activity_features_timeseries_feature_class
from support.frailty_prediction_entry import get_frailty_prediction_entry
from scipy import stats
ExtentedParameters = MinimalFCParameters()
ExtentedParameters['kurtosis'] = None
ExtentedParameters['abs_energy'] = None
ExtentedParameters['skewness'] = None
del ExtentedParameters['length']

class FeatureExtractor:
    def __init__(self):
        self.timestamp_str = 'timestamp'
        self.ecg_heart_rate_str = 'ecg_heart_rate'
        self.ecg_signal_quality_str = 'ecg_signal_quality'
        self.ecg_rr_interval_str = 'ecg_rr_interval'
        self.ecg_heart_rate_variability_str = 'ecg_heart_rate_variability'
        self.acceleration_x_str = 'acceleration_x'
        self.acceleration_y_str = 'acceleration_y'
        self.acceleration_z_str = 'acceleration_z'
        self.health_status_str = "health_status"
        self.participant_id_str = 'participant_id'
        self.feature_id_str = 'feature_id'
        self.breathing_rate_str = "breathing_rate"
        self.activity_class1_str='activity_class1'
        self.activity_class2_str = 'activity_class2'

    def initialize_timestamp_dict(self,attribute_size):

        dataframe= {
            self.participant_id_str: [],
            #self.timestamp_str: [],
            self.ecg_heart_rate_str: [],
            #self.ecg_signal_quality_str : [],
            self.ecg_rr_interval_str: [], self.ecg_heart_rate_variability_str: [],
            self.acceleration_x_str: [], self.acceleration_y_str: [], self.acceleration_z_str: [],
            self.breathing_rate_str:[]
            #,self.health_status_str : []
        }

        if attribute_size.lower()=='full':
            dataframe2={
                self.activity_class1_str:[],
                self.activity_class2_str:[]
            }
        else:
            dataframe2={}

        return dataframe,dataframe2

    def initialize_timestamp_dict_to_aggreation(self,features_names):

        dataframe= {self.participant_id_str: []}
        for feature_name in features_names:
            dataframe[feature_name]=[]

        return dataframe

    def initialize_timestamp_dict_to_activity(self,features_names,activities_list):
        dataframe = {self.participant_id_str: []}
        for activity in activities_list:
            for feature_name in features_names:
                dataframe[feature_name+str(activity)]=[]
        return dataframe

    def clean_attribute_name(self,attribute_name):
        attribute_name_to_insert = re.sub('"', '', attribute_name)
        attribute_name_to_insert = re.sub("'", '', attribute_name_to_insert)
        attribute_name_to_insert = re.sub(",", '_', attribute_name_to_insert)
        attribute_name_to_insert = re.sub('[()]', '_', attribute_name_to_insert)
        attribute_name_to_insert = re.sub(' ', '', attribute_name_to_insert)
        attribute_name_to_insert = re.sub('-', '', attribute_name_to_insert)
        attribute_name_to_insert = ''.join(attribute_name_to_insert.split('.'))

        return attribute_name_to_insert

    def calculate_train_weights(self,datalength_list):

        max_length = max(datalength_list)
        max_index = datalength_list.index(max_length)
        print(max_length,max_index)

        weights=[]

        for data_class in datalength_list:
            weights.append(max_length/data_class)

        return weights

    def minimal_feature_extractor(self,attribute_list):
        attribute_list_mean = float(sum(attribute_list) / len(attribute_list))
        attribute_list__max = max(attribute_list)
        attribute_list__min = min(attribute_list)

        return attribute_list_mean,attribute_list__max,attribute_list__min

    # Define outlier criteria (IQR method)
    def handle_outliers(self,column):
        Q1 = column.quantile(0.25)
        Q3 = column.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (column < lower_bound) | (column > upper_bound)

    def aggregate_features_activity_per_time(self,window_time_name='2m',features_size='Full',aggregate_time_window=1000*60*60*24*5,aggregate_time_window_name='5d',excluded_features=None):
        batch_limit = 300000
        participant_id = None
        participant_features_dict = {}
        ind_timestamp = batch_limit + 1
        activity_name_for_class = 'activity'
        features_list = list(ExtentedParameters.keys())
        aggregation_parameters = {'standard_deviation': None, 'mean': None}
        activities=[0,1,2,3,4,5]

        if excluded_features is not None:
            for excluded_feature in excluded_features:
                features_list.remove(excluded_feature)

        participant_entries = extract_sorted_collection(get_participant_entry_class())
        participant_ids = []
        for participant_entry in participant_entries:
            participant_ids.append(participant_entry.get_participant_id())

        for participant_id in participant_ids:
            timeseries_feature_entries = extract_sorted_collection(
                get_timeseries_feature_class(window_time_name, features_size),
                participant_id_filter=str(participant_id)
            )

            last_activity_class = None
            first_index = True
            last_health_status=None
            features_window_in_activity = 0

            aggregate_dict = {}
            aggregate_timestamp_dict = {}

            for timeseries_feature_entry in timeseries_feature_entries:

                health_status = timeseries_feature_entry.get_health_status()
                participant_id = timeseries_feature_entry.get_participant_id()
                timestamp = timeseries_feature_entry.get_timestamp()
                timeseries_feature_entry.init_features(features_list)

                features_dict = timeseries_feature_entry.get_features_dict()
                activity_class = timeseries_feature_entry.get_activity_class2()

                if first_index:
                    first_index = False
                    first_timestamp_of_day = timestamp
                    last_health_status=health_status
                    data_dtype = self.initialize_timestamp_dict_to_activity(list(features_dict.keys()),activities)

                if last_activity_class is None:
                    last_activity_class = activity_class

                time_diff = int(timestamp) - int(first_timestamp_of_day)

                calculate_feature_flag = time_diff > aggregate_time_window

                if last_health_status!=health_status:
                    print('NEW HEALTH STATUS',last_health_status,health_status)
                    last_health_status=health_status
                    calculate_feature_flag=True

                if calculate_feature_flag:
                    # new activity class - FEATURE EXTRACTION
                    # print('NEW ACTIVTY',activity_class)
                    print('FEATURES IN:', features_window_in_activity, participant_id, first_timestamp_of_day)

                    timeseries_feature_activity_entry = get_activity_features_timeseries_feature_class(aggregate_time_window_name,
                                                                                                       features_size)()

                    timeseries_feature_activity_entry.set_basic(participant_id, first_timestamp_of_day)

                    dfile = pd.DataFrame(data=data_dtype, columns=list(data_dtype.keys()))
                    df_features = extract_features(dfile,
                                                   column_id=self.participant_id_str,
                                                   default_fc_parameters=aggregation_parameters)



                    for attribute_name, attribute_value in df_features.to_dict(orient='records')[0].items():
                        attribute_value_to_insert = attribute_value
                        if is_float(attribute_value):
                            attribute_value_to_insert = round(attribute_value, 5)
                        timeseries_feature_activity_entry.set_attribute(attribute_name, attribute_value_to_insert)
                    timeseries_feature_activity_entry.set_health_status(str(health_status))
                    timeseries_feature_activity_entry.set_activity_class2(activity_class)
                    timeseries_feature_activity_entry.db_write()
                    features_window_in_activity = 0
                    last_activity_class = activity_class

                    data_dtype = self.initialize_timestamp_dict_to_activity(list(features_dict.keys()),activities)
                    first_timestamp_of_day = timestamp
                else:
                    features_window_in_activity = features_window_in_activity + 1

                data_dtype[self.participant_id_str] = participant_id
                for feature_id in features_dict.keys():
                    for activity_class_selected in activities:
                        if activity_class==activity_class_selected:
                            value_to_insert=features_dict[feature_id]


                            data_dtype[feature_id+str(activity_class)].append(float(value_to_insert))
                        else:
                            data_dtype[feature_id + str(activity_class_selected)].append(float(0))

            # the last activity of Participant

            if first_index:
                continue

            timeseries_feature_activity_entry = get_activity_features_timeseries_feature_class(aggregate_time_window_name,
                                                                                               features_size)()
            timeseries_feature_activity_entry.set_basic(participant_id, first_timestamp_of_day)


            dfile = pd.DataFrame(data=data_dtype, columns=list(data_dtype.keys()))

            df_features = extract_features(dfile,
                                           column_id=self.participant_id_str,
                                           default_fc_parameters=aggregation_parameters)

            for attribute_name, attribute_value in df_features.to_dict(orient='records')[0].items():
                attribute_value_to_insert = attribute_value
                if is_float(attribute_value):
                    attribute_value_to_insert = round(attribute_value, 5)
                timeseries_feature_activity_entry.set_attribute(attribute_name, attribute_value_to_insert)
            timeseries_feature_activity_entry.set_health_status(str(health_status))
            timeseries_feature_activity_entry.set_activity_class2(activity_class)
            timeseries_feature_activity_entry.db_write()

    def aggregate_features_per_activity(self,window_time_name='2m',features_size='Full'):
        batch_limit = 300000
        participant_id = None
        participant_features_dict = {}
        ind_timestamp = batch_limit + 1
        activity_name_for_class = 'activity'
        features_list = ExtentedParameters.keys()
        aggregation_parameters={'mean': None, 'standard_deviation':None, 'minimum':None , 'maximum':None}

        participant_entries=extract_sorted_collection(get_participant_entry_class())
        participant_ids=[]
        for participant_entry in participant_entries:
            participant_ids.append(participant_entry.get_participant_id())


        for participant_id in participant_ids:
            timeseries_feature_entries = extract_sorted_collection(
                get_timeseries_feature_class(window_time_name, features_size),
                participant_id_filter=str(participant_id)
            )

            last_activity_class=None
            first_index = True
            features_window_in_activity=0

            aggregate_dict={}
            aggregate_timestamp_dict = {}

            for timeseries_feature_entry in timeseries_feature_entries:

                health_status = timeseries_feature_entry.get_health_status()
                participant_id = timeseries_feature_entry.get_participant_id()
                timestamp = timeseries_feature_entry.get_timestamp()

                timeseries_feature_entry.init_features(features_list)

                features_dict=timeseries_feature_entry.get_features_dict()
                activity_class = timeseries_feature_entry.get_activity_class2()

                if first_index:
                    first_index=False
                    first_timestamp_of_day=timestamp
                    data_dtype = self.initialize_timestamp_dict_to_aggreation(list(features_dict.keys()))

                if last_activity_class is None:
                    last_activity_class=activity_class


                if activity_class!=last_activity_class:
                    #new activity class - FEATURE EXTRACTION
                    #print('NEW ACTIVTY',activity_class)
                    print('FEATURES IN:',features_window_in_activity,participant_id, first_timestamp_of_day)

                    timeseries_feature_activity_entry = get_activity_features_timeseries_feature_class(window_time_name,
                                                                            features_size)()

                    timeseries_feature_activity_entry.set_basic(participant_id, first_timestamp_of_day)

                    dfile = pd.DataFrame(data=data_dtype, columns=list(data_dtype.keys()))

                    df_features = extract_features(dfile,
                                                   column_id=self.participant_id_str,
                                                   default_fc_parameters=aggregation_parameters)

                    for attribute_name, attribute_value in df_features.to_dict(orient='records')[0].items():
                        attribute_value_to_insert = attribute_value
                        if is_float(attribute_value):
                            attribute_value_to_insert = round(attribute_value, 5)
                        timeseries_feature_activity_entry.set_attribute(attribute_name,attribute_value_to_insert)
                    timeseries_feature_activity_entry.set_health_status(str(health_status))
                    timeseries_feature_activity_entry.set_activity_class2(activity_class)
                    timeseries_feature_activity_entry.db_write()
                    features_window_in_activity=0
                    last_activity_class=activity_class

                    data_dtype = self.initialize_timestamp_dict_to_aggreation(list(features_dict.keys()))
                    first_timestamp_of_day = timestamp
                else:
                    features_window_in_activity=features_window_in_activity+1

                data_dtype[self.participant_id_str] = participant_id
                for feature_id in features_dict.keys():
                    data_dtype[feature_id].append(features_dict[feature_id])

            #the last activity of Participant

            if first_index:
                continue

            timeseries_feature_activity_entry = get_activity_features_timeseries_feature_class(window_time_name,
                                                                    features_size)()
            timeseries_feature_activity_entry.set_basic(participant_id, first_timestamp_of_day)
            dfile = pd.DataFrame(data=data_dtype, columns=list(data_dtype.keys()))
            df_features = extract_features(dfile,
                                           column_id=self.participant_id_str,
                                           default_fc_parameters=aggregation_parameters)

            for attribute_name, attribute_value in df_features.to_dict(orient='records')[0].items():
                attribute_value_to_insert = attribute_value
                if is_float(attribute_value):
                    attribute_value_to_insert = round(attribute_value, 5)
                timeseries_feature_activity_entry.set_attribute(attribute_name,attribute_value_to_insert)
            timeseries_feature_activity_entry.set_health_status(str(health_status))
            timeseries_feature_activity_entry.set_activity_class2(activity_class)
            timeseries_feature_activity_entry.db_write()

    def extract_features_from_smartwatch_dataset(self,
                                                 device_source='wwsx',
                                                 skip_multi_health_status_participants=False,
                                                 window_time_size=1000 * 60 * 60 * 24,
                                                 window_time_name='24hr',
                                                 series_size=None,
                                                 series_size_name=None,
                                                 features_size='Basic',
                                                 attribute_size='Basic',
                                                 features_per_activity=False
                                                 ):

        data_dtype,activity_data_dtype = self.initialize_timestamp_dict(attribute_size)

        if series_size is not None:
            window_time_name = series_size_name

        participant_entries = batch_db_read(get_participant_entry_class())

        data_dtype_features = None

        settings = MinimalFCParameters()

        if features_size.capitalize() == 'Extended' or features_size.capitalize()=='Full':
            settings = ExtentedParameters

        if features_size.lower() == 'full':
            pass
            #settings['autocorrelation'] = [{'lag': 0}]

        participant_ids_dict = {'Frail': [], 'Pre-frail': [], 'Non-frail': []}

        if not skip_multi_health_status_participants:
            participant_ids_dict['Multi-Health-Status'] = []

        for participant_entry in participant_entries:
            participant_id = participant_entry.get_participant_id()

            participant_frailty_status_list = participant_entry.get_participant_frailty_status()

            if skip_multi_health_status_participants:
                if len(participant_frailty_status_list) > 1:
                    continue

                health_status = participant_frailty_status_list[0]['frailty_status']
                participant_ids_dict[health_status].append(participant_id)
            else:
                if len(participant_frailty_status_list) > 1:
                    health_status = 'Multi-Health-Status'
                else:
                    health_status = participant_frailty_status_list[0]['frailty_status']
                participant_ids_dict[health_status].append(participant_id)

        if skip_multi_health_status_participants:
            data_length = [len(participant_ids_dict['Frail']), len(participant_ids_dict['Pre-frail']),
                           len(participant_ids_dict['Non-frail'])]
            train_weights = self.calculate_train_weights(data_length)
        batch_limit = 500000
        minimum_samples_per_window=125 # 5 seconds

        for health_status_id, participant_ids_list in participant_ids_dict.items():
            print(health_status_id)

            for participant_id in participant_ids_list:
                participant_timestamp_class = get_participant_timestamp_entry_class(device_source, participant_id)
                participant_data_index = 0
                print('Participant ID', participant_id)

                if participant_id != '2109':
                    pass

                daily_timestamp_index = 0
                ind_timestamp = batch_limit + 1
                last_timestamp = None
                first_timestamp_of_day = None
                health_status = health_status_id

                while ind_timestamp >= batch_limit:
                    print('NEW BATCH-', last_timestamp)
                    participant_timestamp_entries = extract_sorted_collection(participant_timestamp_class,
                                                                              limit=batch_limit,
                                                                              low_timestamp_threshold=last_timestamp)

                    ind_timestamp = 0
                    last_activity_class=None

                    deleted_timestamps=0

                    for participant_timestamp_entry in participant_timestamp_entries:
                        participant_data_index = participant_data_index + 1

                        timestamp = participant_timestamp_entry.get_timestamp()
                        activity_class1 = participant_timestamp_entry.get_activity_class1()
                        activity_class2 = participant_timestamp_entry.get_activity_class2()
                        ecg_signal_quality = participant_timestamp_entry.get_ecg_signal_quality()
                        ecg_heart_rate = participant_timestamp_entry.get_ecg_heart_rate()
                        if ecg_signal_quality < 50:
                            # metric is unreliable
                            deleted_timestamps=deleted_timestamps+1
                            continue
                        elif (ecg_heart_rate<30 and ecg_signal_quality<150) or (ecg_signal_quality < 90 and ecg_heart_rate<40) :
                            # metric is unreliable
                            deleted_timestamps = deleted_timestamps + 1
                            continue

                        if last_activity_class is None:
                            last_activity_class = activity_class2

                        if first_timestamp_of_day is None:
                            first_timestamp_of_day = timestamp

                        last_timestamp = timestamp

                        time_diff = int(timestamp) - int(first_timestamp_of_day)

                        if series_size is None:
                            calculate_feature_flag = time_diff > window_time_size
                        else:
                            calculate_feature_flag = participant_data_index == series_size

                        if last_activity_class != activity_class2 and features_per_activity:
                            #new activity
                            calculate_feature_flag=True
                            last_activity_class=activity_class2

                        if calculate_feature_flag:
                            print('EXTRACT FEATURES', ind_timestamp, first_timestamp_of_day, timestamp, daily_timestamp_index,deleted_timestamps)

                            participant_data_index = 0
                            deleted_timestamps=0

                            dfile = pd.DataFrame(data=data_dtype, columns=list(data_dtype.keys()))

                            df_features = extract_features(dfile,
                                                           column_id=self.participant_id_str,
                                                           default_fc_parameters=settings)

                            timeseries_feature_entry = get_timeseries_feature_class(window_time_name,
                                                                                    features_size)()

                            timeseries_feature_entry.set_basic(participant_id, first_timestamp_of_day)

                            for attribute_name, attribute_value in df_features.to_dict(orient='records')[0].items():

                                attribute_name_to_insert = self.clean_attribute_name(attribute_name)
                                attribute_value_to_insert = attribute_value
                                if is_float(attribute_value):
                                    attribute_value_to_insert = round(attribute_value, 5)
                                timeseries_feature_entry.set_attribute(attribute_name_to_insert, attribute_value_to_insert)
                            timeseries_feature_entry.set_health_status(str(health_status))
                            timeseries_feature_entry.set_activity_class2(activity_class2)

                            if daily_timestamp_index > minimum_samples_per_window:
                                timeseries_feature_entry.db_write()

                            data_dtype,activity_data_dtype = self.initialize_timestamp_dict(attribute_size)
                            daily_timestamp_index = 0

                            first_timestamp_of_day = timestamp

                        ecg_heart_rate = participant_timestamp_entry.get_ecg_heart_rate()
                        ecg_signal_quality = participant_timestamp_entry.get_ecg_signal_quality()
                        ecg_rr_interval = participant_timestamp_entry.get_ecg_rr_interval()
                        ecg_heart_rate_variability = participant_timestamp_entry.get_ecg_heart_rate_variability()
                        acceleration_x = participant_timestamp_entry.get_acceleration_x()
                        acceleration_y = participant_timestamp_entry.get_acceleration_y()
                        acceleration_z = participant_timestamp_entry.get_acceleration_z()
                        health_status = participant_timestamp_entry.get_health_status()
                        breathing_rate = participant_timestamp_entry.get_breathing_rate()
                        activity_class1 = participant_timestamp_entry.get_activity_class1()
                        activity_class2 = participant_timestamp_entry.get_activity_class2()

                        data_dtype[self.participant_id_str].append(participant_id)
                        # data_dtype[self.timestamp_str].append(float(timestamp))
                        data_dtype[self.ecg_heart_rate_str].append(float(ecg_heart_rate))
                        #data_dtype[self.ecg_signal_quality_str].append(float(ecg_signal_quality))
                        data_dtype[self.ecg_rr_interval_str].append(float(ecg_rr_interval))
                        data_dtype[self.ecg_heart_rate_variability_str].append(float(ecg_heart_rate_variability))

                        data_dtype[self.acceleration_x_str].append(float(acceleration_x))
                        data_dtype[self.acceleration_y_str].append(float(acceleration_y))
                        data_dtype[self.acceleration_z_str].append(float(acceleration_z))
                        data_dtype[self.breathing_rate_str].append(float(breathing_rate))


                        if attribute_size.lower()=='full':
                            #activity_data_dtype[self.activity_class1_str].append(int(activity_class1))
                            #activity_data_dtype[self.activity_class2_str].append(int(activity_class2))
                            pass

                        ind_timestamp = ind_timestamp + 1
                        daily_timestamp_index = daily_timestamp_index + 1

                    if ind_timestamp == batch_limit:
                        continue

                    print(ind_timestamp, first_timestamp_of_day, last_timestamp, daily_timestamp_index)
                    dfile = pd.DataFrame(data=data_dtype, columns=list(data_dtype.keys()))

                    df_features = extract_features(dfile, column_id=self.participant_id_str,
                                                   default_fc_parameters=settings)

                    if data_dtype_features is None:
                        data_dtype_features = pd.DataFrame(data=df_features, columns=list(df_features.columns))
                    else:
                        data_dtype_features = data_dtype_features.append(df_features)

                    timeseries_feature_entry = get_timeseries_feature_class(window_time_name,
                                                                            features_size)()

                    timeseries_feature_entry.set_basic(participant_id, first_timestamp_of_day)

                    attribute_counter = 0
                    for attribute_name, attribute_value in df_features.to_dict(orient='records')[0].items():
                        attribute_name_to_insert = self.clean_attribute_name(attribute_name)
                        attribute_value_to_insert = attribute_value
                        attribute_counter = attribute_counter + 1
                        if is_float(attribute_value):
                            attribute_value_to_insert = round(attribute_value, 5)
                        timeseries_feature_entry.set_attribute(attribute_name_to_insert, attribute_value_to_insert)
                    timeseries_feature_entry.set_health_status(health_status)
                    timeseries_feature_entry.set_activity_class2(activity_class2)

                    if daily_timestamp_index >= minimum_samples_per_window:
                        timeseries_feature_entry.db_write()

                data_dtype,activity_data_dtype = self.initialize_timestamp_dict(attribute_size)

def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

async def async_main():

    s = time.perf_counter()

    plugin = FeatureExtractor()

    plugin.extract_features_from_smartwatch_dataset(
        features_size='full',
        window_time_size=1000 * 60 * 60 * 24*7,
        window_time_name='7d',
        attribute_size='full',
        features_per_activity=True
    )


    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.2f} seconds.")

def main():
    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_until_complete(async_main())
