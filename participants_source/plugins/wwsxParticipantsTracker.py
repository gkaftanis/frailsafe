from participants_source.participants_source_base import ParticipantsTrackerBase
from support.participant_entry import get_participant_entry_class,\
    get_participant_timestamp_entry_class,ParticipantFrailtyStatusEntry
from support.connect_with_db import batch_db_write,read_collection_keys,batch_db_read
import settings

import time
import datetime
import os
import glob
from matplotlib import pyplot as plt
from scipy.io import loadmat


wwsx_folder_path='data/wwsx_matlab'
device_source='wwsx'

class WwsxParticipantsTracker(ParticipantsTrackerBase):
    def __init__(self,device_source):

        self.device_source=device_source
        self.time_date_format = "%d/%m/%Y %H:%M:%S.%f"
        self.ts_date_str = "ts_date"
        self.ecg_rr_str = "ecg_rr"
        self.ecg_quality_str = "ecg_quality"
        self.ecg_hr_str = "ecg_hr"
        self.ecg_hrv = "ecg_hrv"
        self.acc_x_str = "acc_x"
        self.acc_y_str = "acc_y"
        self.acc_z_str = "acc_z"
        self.act_class1 = "act_class1"
        self.act_class2 = "act_class2"
        self.ba_str = "ba"
        self.br_str = "br"
        self.resp_piezo_str = "resp_piezo"

    async def write_frailty_status_to_db(self):

        participant_entry_dict ={ }

        for health_status in settings.data_subfolder_list:
            subfolder_path = health_status

            mat_files = glob.glob(wwsx_folder_path + '/' + subfolder_path + '/*.mat')

            index_file = 0

            # Iterate mat files
            for fname in mat_files:
                index_file = index_file + 1

                # Load mat file data into data.
                participant_data = loadmat(fname)

                partictipant_index = 0

                if index_file==20:
                    pass

                paricipant_entry_class = get_participant_entry_class()
                participant_entry = paricipant_entry_class()
                participant_id = participant_data['part_id'][0]

                if participant_id in participant_entry_dict.keys():
                    participant_entry = participant_entry_dict[participant_id]

                current_timestamp_date = participant_data['ts_time'][0]
                current_date = participant_data[self.ts_date_str][partictipant_index]

                start_timestamp = self.date_to_epoch(current_date,current_timestamp_date,self.time_date_format)

                end_timestamp_date = participant_data['ts_time'][-1]
                end_date = participant_data[self.ts_date_str][-1]

                end_timestamp = self.date_to_epoch(end_date, end_timestamp_date, self.time_date_format)

                participant_frailty_status_entry = ParticipantFrailtyStatusEntry()
                participant_frailty_status_entry.set_frailty_status(health_status)
                participant_frailty_status_entry.set_frailty_status_start_timestamp(start_timestamp)
                participant_frailty_status_entry.set_frailty_status_end_timestamp(end_timestamp)

                participant_entry.set_participant_id(str(participant_id))
                participant_entry.add_participant_frailty_status(participant_frailty_status_entry)


                participant_entry_dict[participant_id] = participant_entry


        for participant_entry in participant_entry_dict.values():
            participant_entry_preprocessed = participant_entry
            participant_frailty_status = participant_entry_preprocessed.get_participant_frailty_status()

            participant_frailty_status.sort(key=lambda x: x.frailty_status_start_timestamp)

            if len(participant_frailty_status)>1:
                last_frailty_status_entry = participant_frailty_status[0]

                participant_frailty_status_entry_list = [last_frailty_status_entry]
                status_index = 0

                modify_flag = False
                for part_status in participant_frailty_status:
                    status_index = status_index + 1
                    if part_status.get_frailty_status()==last_frailty_status_entry.get_frailty_status():
                        participant_frailty_status_entry_list[-1].set_frailty_status_end_timestamp(
                            part_status.get_frailty_status_end_timestamp())
                        modify_flag = True
                    else:
                        participant_frailty_status_entry_list.append(part_status)

                    last_frailty_status_entry = part_status

                if modify_flag:

                    participant_entry_preprocessed.set_participant_frailty_status_list([])

                    for participant_frailty_timestamp_modified in participant_frailty_status_entry_list:
                        participant_entry_preprocessed.add_participant_frailty_status(
                            participant_frailty_timestamp_modified.__dict__)

                participant_entry_preprocessed.db_write()

    async def read_participants_data(self):

        paricipant_id_stored = read_collection_keys('ParticipantEntry')
        frail_status = {}

        for health_status in settings.data_subfolder_list:
            subfolder_path = health_status

            mat_files = glob.glob(wwsx_folder_path+'/' + subfolder_path+'/*.mat')

            index_file = 0

            # Iterate mat files
            for fname in mat_files:

                index_file = index_file + 1

                # Load mat file data into data.
                participant_data = loadmat(fname)

                partictipant_index = 0

                participant_entry_class = get_participant_entry_class()
                paricipant_entry = participant_entry_class()
                participant_id = participant_data['part_id'][0]

                current_timestamp_date = participant_data['ts_time'][0]
                current_date = participant_data[self.ts_date_str][partictipant_index]

                timestamp_format = current_date[6:8] + '/' + current_date[4:6] + '/' + current_date[:4]
                timestamp_formatted = timestamp_format + ' ' + current_timestamp_date
                timestamp_epoch = datetime.datetime.strptime(timestamp_formatted, self.time_date_format).timestamp()
                timestamp_epoch_in_milliseconds = int(timestamp_epoch * 1000)
                start_timesamp = timestamp_epoch_in_milliseconds
                print(participant_id)

                participant_frailty_status_entry = ParticipantFrailtyStatusEntry()
                participant_frailty_status_entry.set_frailty_status(health_status)
                participant_frailty_status_entry.set_frailty_status_start_timestamp(start_timesamp)
                #participant_frailty_status_entry.set_frailty_status_end_timestamp(end_timestamp)
                print(participant_frailty_status_entry.__dict__)

                if participant_id in frail_status.keys():
                    if health_status not in frail_status[participant_id]:
                        frail_status[participant_id].append(health_status)
                else:
                    frail_status[participant_id] = [health_status]

                #if participant_id in paricipant_id_stored:
                #    continue

                paricipant_entry.set_participant_id(str(participant_id))
                paricipant_entry.add_participant_frailty_status( participant_frailty_status_entry.__dict__ )

                #paricipant_entry.db_write()

                for timepoint in participant_data['ts_time']:
                    current_timestamp_date = timepoint
                    current_date = participant_data[self.ts_date_str][partictipant_index]

                    timestamp_format = current_date[6:8]+'/'+current_date[4:6]+'/'+current_date[:4]
                    timestamp_formatted = timestamp_format + ' ' + current_timestamp_date
                    timestamp_epoch = datetime.datetime.strptime(timestamp_formatted, self.time_date_format).timestamp()
                    timestamp_epoch_in_milliseconds = int(timestamp_epoch*1000)

                    ecg_rr = participant_data[self.ecg_rr_str][0][partictipant_index]
                    #ECG signal quality (value: 0-255 where 0=poor and 255=excellent - sampling rate: 1/5 sec)
                    ecg_quality = participant_data[self.ecg_quality_str][0][partictipant_index]
                    #Heart rate (value: Beats/minute - sampling rate: 1/5 sec)
                    ecg_hr = participant_data[self.ecg_hr_str][0][partictipant_index]
                    #Heart rate variability (value: ms - sampling rate: 1/60 sec)
                    ecg_hrv = participant_data[self.ecg_hrv][0][partictipant_index]

                    acc_x = participant_data[self.acc_x_str][0][partictipant_index]
                    acc_y = participant_data[self.acc_y_str][0][partictipant_index]
                    acc_z = participant_data[self.acc_z_str][0][partictipant_index]

                    #Activity performed (value: 0-4 where 0=other, 1=lying, 2=standing/sitting,
                    # 3=walking and 4=running - sampling rate: 1/5 sec)
                    act_class1 = participant_data[self.act_class1][0][partictipant_index]
                    act_class2 = participant_data[self.act_class2][0][partictipant_index]
                    act_class1 = int(act_class1)
                    act_class2 = int(act_class2)

                    #Breathing Amplitude (value: logic levels - sampling rate: 1/15 sec)
                    ba = participant_data[self.ba_str][0][partictipant_index]

                    #Breathing rate (value: Breaths/minute - sampling rate: 1/5 sec)
                    br = participant_data[self.br_str][0][partictipant_index]

                    #Electric signal measuring the chest pressure on the piezoelectric point
                    # (value: 0.8 mV - sampling rate: 25 Hz)
                    resp_piezo = participant_data[self.resp_piezo_str][0][partictipant_index]
                    resp_piezo = float(resp_piezo)

                    participant_timestamp_entry_class = get_participant_timestamp_entry_class(device_source,
                                                                                              participant_id)
                    participant_timestamp_entry = participant_timestamp_entry_class()
                    participant_timestamp_entry.set_timestamp(str(int(timestamp_epoch_in_milliseconds)))
                    participant_timestamp_entry.set_ecg_heart_rate(float(ecg_hr))
                    participant_timestamp_entry.set_ecg_signal_quality(float(ecg_quality))
                    participant_timestamp_entry.set_ecg_rr_interval(float(ecg_rr))
                    participant_timestamp_entry.set_ecg_heart_rate_variability(float(ecg_hrv))
                    participant_timestamp_entry.set_acceleration_x(float(acc_x))
                    participant_timestamp_entry.set_acceleration_y(float(acc_y))
                    participant_timestamp_entry.set_acceleration_z(float(acc_z))

                    participant_timestamp_entry.set_breathing_rate(float(br))
                    participant_timestamp_entry.set_breathing_amplitude(float(ba))

                    participant_timestamp_entry.set_resp_piezoelectric(float(resp_piezo))
                    participant_timestamp_entry.set_activity_class1(float(act_class1))
                    participant_timestamp_entry.set_activity_class2(float(act_class2))
                    participant_timestamp_entry.set_health_status(health_status)

                    partictipant_index = partictipant_index + 1

                    #paricipant_entry.add_timestamp_entry(dict(
                    #    participant_timestamp_entry.__dict__))

                    participant_timestamp_entry.db_write()


def get_plugin_class():
    return WwsxParticipantsTracker

async def main_for_testing():
    plugin = WwsxParticipantsTracker('wwsx')
    s = time.perf_counter()
    #await plugin.write_frailty_status_to_db()
    #await plugin.read_participants_data()


    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.2f} seconds.")

def main():
   import asyncio
   loop = asyncio.get_event_loop()
   loop.run_until_complete(main_for_testing())
