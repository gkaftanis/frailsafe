from support.connect_with_db import ConnectwithDB


class TimeseriesFeaturesEntry(ConnectwithDB):
    @classmethod
    def get_key_value(cls):
        return 'features_id'

    def set_basic(self,participant_id,timestamp):
        self.participant_id=participant_id
        self.timestamp=timestamp
        self.features_id=participant_id+'_'+timestamp

    def init_features(self,features_list=None,aggregated_features=None):
        self.default_attributes= ['acceleration_x','acceleration_y','acceleration_z',
                                  'breathing_rate','ecg_heart_rate_variability',
                                  'ecg_rr_interval','ecg_heart_rate']
        if features_list is None:
            self.features_list = [
                'sum_values', 'median', 'mean','standard_deviation',
                'variance','maximum','minimum','kurtosis','abs_energy','skewness'
            ]
        else:
            self.features_list=features_list

        if aggregated_features is None:
            self.aggregated_features=['mean','standard_deviation']
        else:
            self.aggregated_features=aggregated_features

    def set_timestamp(self,timestamp):
        #in milliseconds
        self.timestamp=timestamp

    def set_attribute(self,attribute_name,attribute_value):
        setattr(self,attribute_name, attribute_value)

    def set_health_status(self,health_status):
        self.health_status = health_status

    def set_ecg_heart_rate(self,ecg_heart_rate):
        self.ecg_heart_rate = ecg_heart_rate

    def set_ecg_signal_quality(self,ecg_signal_quality):
        self.ecg_signal_quality = ecg_signal_quality

    def set_ecg_rr_interval(self,ecg_rr_interval):
        self.ecg_rr_interval = ecg_rr_interval

    def set_ecg_heart_rate_variability(self,ecg_heart_rate_variability):
        self.ecg_heart_rate_variability = ecg_heart_rate_variability

    def set_acceleration_x(self,acceleration_x):
        self.acceleration_x = acceleration_x

    def set_acceleration_y(self,acceleration_y):
        self.acceleration_y = acceleration_y

    def set_acceleration_z(self,acceleration_z):
        self.acceleration_z = acceleration_z

    def set_breathing_rate(self,breathing_rate):
        self.breathing_rate = breathing_rate

    def set_breathing_amplitude(self,breathing_amplitude):
        self.breathing_amplitude = breathing_amplitude

    def set_resp_piezoelectric(self,resp_piezoelectric):
        self.resp_piezoelectric = resp_piezoelectric

    def set_activity_class1(self,activity_class1):
        self.activity_class1 = activity_class1

    def set_activity_class2(self,activity_class2):
        self.activity_class2 = activity_class2

    def get_timestamp(self):
        return self.timestamp

    def get_ecg_heart_rate_features(self):
        features_list = []
        attribute_name = 'ecg_heart_rate'

        for feature in self.features_list:
            features_list.append(getattr(self,attribute_name+'__'+feature,))

        return features_list

    def get_ecg_heart_rate(self):
        return self.ecg_heart_rate

    def get_ecg_signal_quality_features(self):
        features_list = []
        attribute_name = 'ecg_rr_interval'

        for feature in self.features_list:
            features_list.append(getattr(self,attribute_name+'__'+feature,))

        return features_list

    def get_ecg_signal_quality(self):
        return self.ecg_signal_quality

    def get_ecg_rr_interval_features(self):
        features_list = []
        attribute_name = 'ecg_rr_interval'

        for feature in self.features_list:
            features_list.append(getattr(self,attribute_name+'__'+feature,))

        return features_list

    def get_ecg_rr_interval(self):
        return self.ecg_rr_interval

    def get_ecg_heart_rate_variability_features(self):
        features_list = []
        attribute_name = 'ecg_heart_rate_variability'

        for feature in self.features_list:
            features_list.append(getattr(self,attribute_name+'__'+feature,))

        return features_list

    def get_ecg_heart_rate_variability(self):
        return self.ecg_heart_rate_variability

    def get_acceleration_x_features(self):
        features_list = []
        attribute_name = 'acceleration_x'

        for feature in self.features_list:
            features_list.append(getattr(self,attribute_name+'__'+feature,))

        return features_list

    def get_acceleration_x(self):
        return self.acceleration_x

    def get_features_dict(self,for_aggregated=False,get_keys_only=False):
        if for_aggregated:
            sep='_'
        else:
            sep='__'
        features_dict={}
        for attribute_name in self.default_attributes:
            for feature in self.features_list:
                if get_keys_only:
                    features_dict[attribute_name + '_' + feature] = []
                else:
                    features_dict[attribute_name + '_' + feature] = getattr(self, attribute_name + sep + feature, )

        return features_dict


    def get_acceleration_y_features(self):
        features_list = []
        attribute_name = 'acceleration_y'

        for feature in self.features_list:
            features_list.append(getattr(self,attribute_name+'__'+feature,))

        return features_list

    def get_acceleration_y(self):
        return self.acceleration_y

    def get_acceleration_z_features(self):
        features_list = []
        attribute_name = 'acceleration_z'

        for feature in self.features_list:
            features_list.append(getattr(self,attribute_name+'__'+feature,))

        return features_list

    def get_acceleration_z(self):
        return self.acceleration_z

    def get_breathing_rate_features(self):
        features_list = []
        attribute_name = 'breathing_rate'

        for feature in self.features_list:
            features_list.append(getattr(self,attribute_name+'__'+feature,))

        return features_list

    def get_activity_class1_features(self):
        features_list = []
        attribute_name = 'activity_class1'

        for feature in self.features_list:
            try:
                act_class_feature=getattr(self,attribute_name+'__'+feature,)
                features_list.append(act_class_feature)
            except AttributeError:
                pass

        return features_list

    def get_activity_class2_features(self):
        features_list = []
        attribute_name = 'activity_class2'

        for feature in self.features_list:
            try:
                act_class_feature = getattr(self, attribute_name + '__' + feature, )
                features_list.append(act_class_feature)
            except AttributeError:
                pass

        return features_list

    def get_aggregated_features(self,activity_class_list):
        features_dict = self.get_features_dict(True,True)
        aggregated_features_list=[]
        for activity in activity_class_list:
            for feature_id in features_dict.keys():
                for aggr_features in self.aggregated_features:
                    act_class_feature = getattr(self, feature_id + str(activity)+'__' + aggr_features, )
                    aggregated_features_list.append(act_class_feature)

        return aggregated_features_list


    def get_breathing_rate(self):
        return self.breathing_rate

    def get_breathing_amplitude(self):
        return self.breathing_amplitude

    def get_resp_piezoelectric(self):
        return self.resp_piezoelectric

    def get_activity_class1(self):
        return self.activity_class1

    def get_activity_class2(self):
        return self.activity_class2

    def get_health_status(self):
        return self.health_status

    def get_participant_id(self):
        return self.participant_id

    def get_attribute(self,attribute_name):
        return getattr(self, attribute_name)

    def get_feature(self,attribute_name,feature_name):
        return getattr(self, attribute_name+'__'+feature_name)



def get_timeseries_feature_class(time_window,features_size):
    return type('Timeseries'+time_window+features_size.capitalize()+'FeaturesEntry',(TimeseriesFeaturesEntry,),{})

def get_activity_features_timeseries_feature_class(time_window,features_size):
    return type('Timeseries'+time_window+features_size.capitalize()+'AggregatedActivityFeaturesEntry',(TimeseriesFeaturesEntry,),{})