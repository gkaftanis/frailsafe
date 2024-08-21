
import datetime


class ParticipantsTrackerBase:
    async def read_participants_data(self):
        raise Exception('implement me')

    def date_to_epoch(self, current_date, current_timestamp_date,time_date_format):
        timestamp_format = current_date[6:8] + '/' + current_date[4:6] + '/' + current_date[:4]
        timestamp_formatted = timestamp_format + ' ' + current_timestamp_date
        timestamp_epoch = datetime.datetime.strptime(timestamp_formatted, time_date_format).timestamp()
        timestamp_epoch_in_milliseconds = int(timestamp_epoch * 1000)
        timestamp = timestamp_epoch_in_milliseconds

        return int(timestamp)


