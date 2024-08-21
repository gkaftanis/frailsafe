from support.connect_with_db import ConnectwithDB

class ParticipantEntry(ConnectwithDB):
    @classmethod
    def get_key_value(cls):
        return 'participant_id'

    def set_participant_id(self,participant_id):
        self.participant_id = participant_id

    def set_health_status(self,health_status):
        self.health_status = health_status

    def add_participant_frailty_status(self,participant_frailty_status):
        try:
            self.frailty_status.append(participant_frailty_status)
        except:
            self.frailty_status=[participant_frailty_status]

    def set_participant_frailty_status_list(self, participant_frailty_status_list):
        self.frailty_status = participant_frailty_status_list

    def get_participant_id(self):
        return self.participant_id

    def get_participant_frailty_status(self):
        return self.frailty_status

    def get_timestamp_entry(self):
        return timestamp_entry


class ParticipantFrailtyStatusEntry(ConnectwithDB):
    @classmethod
    def get_key_value(cls):
        return 'frailty_status_start_timestamp'

    def set_frailty_status(self,frailty_status):
        self.frailty_status = frailty_status

    def set_frailty_status_start_timestamp(self,frailty_status_start_timestamp):
        self.frailty_status_start_timestamp = frailty_status_start_timestamp

    def set_frailty_status_end_timestamp(self,frailty_status_end_timestamp):
        self.frailty_status_end_timestamp = frailty_status_end_timestamp

    def get_frailty_status(self):
        return self.frailty_status

    def get_frailty_status_start_timestamp(self):
        return self.frailty_status_start_timestamp

    def get_frailty_status_end_timestamp(self):
        return self.frailty_status_end_timestamp


class ParticipantTimestamp(ConnectwithDB):
    @classmethod
    def get_key_value(cls):
        return 'timestamp'

    def set_timestamp(self,timestamp):
        #in milliseconds
        self.timestamp=timestamp

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

    def get_ecg_heart_rate(self):
        return self.ecg_heart_rate

    def get_ecg_signal_quality(self):
        return self.ecg_signal_quality

    def get_ecg_rr_interval(self):
        return self.ecg_rr_interval

    def get_ecg_heart_rate_variability(self):
        return self.ecg_heart_rate_variability

    def get_acceleration_x(self):
        return self.acceleration_x

    def get_acceleration_y(self):
        return self.acceleration_y

    def get_acceleration_z(self):
        return self.acceleration_z

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

def get_participant_entry_class():
    return type('ParticipantEntry',(ParticipantEntry,),{})

def get_participant_timestamp_entry_class(device_source,participant_id):
    return type(device_source.capitalize()+ participant_id.capitalize()
                +'ParticipantTimestamp',(ParticipantTimestamp,),{})

