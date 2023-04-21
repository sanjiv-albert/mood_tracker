import datetime
import os
import pickle

import analysis
from Data_Preprocessing import DataCreator
from XML_extraction import HeartRate, BloodOxygen


def test_main():
    print('Test getting raw data:')
    data = analysis.get_raw_data()
    # print most recent 10 heart rate values
    for i in range(10):
        print(data[i])

    print('\nTest filtering data by source name:')
    sanji_data = analysis.filter_data_by_source_name(data)
    # print most recent 10 heart rate values
    for i in range(10):
        print(sanji_data[i])
    data = sanji_data
    data = [x for x in data if type(x) == HeartRate]
    print('\nTest finding heart rate averages by motion context:')
    averages, _ = analysis.find_average_heartrate_by_motion_context(data)
    # Print averages from lowest motion context to highest motion context
    for key in sorted(averages.keys()):
        print(f'{key}: {averages[key]}')

    print('\nTest finding heart rate averages by time of day:')
    averages, _ = analysis.find_average_heartrate_by_time_of_day(data)
    # Print averages from 0 to 23
    for key in sorted(averages.keys()):
        print(f'{key}: {averages[key]}')

    print('\nTest finding mean, median, and mode:')
    mean, median, mode = analysis.mean_median_mode(data)
    print(f'Mean: {mean}\nMedian: {median}\nMode: {mode}')
    print('Testing mean, median, and mode on just sedentary:')
    sendentary_data = analysis.get_sedentary_data(data)
    mean, median, mode = analysis.mean_median_mode(sendentary_data)
    print(f'Mean: {mean}\nMedian: {median}\nMode: {mode}')


def get_mood_data():
    # Array of tuples ((mood, datetime(start), datetime(end)), ...)
    # moods are happiness, sadness, anger, fear, disgust and neutrality.
    # represnted as chars 'h', 's', 'a', 'f', 'd', 'n'

    # if no year, month, or day is specified, it is assumed to be 2023-4-20
    # data: (sad, 2023-4-20 9:08:00am, 2023-4-20 9:10:00am), (happy, 2023-4-20 11:33am, 2023-4-20 11:40am), (happy, 2023-4-20 11:46:00pm, 2023-4-21 12:09:00am)
    # (h, 9:44pm, 9:51pm), (h, 11:08pm, 11:24pm), (n, 9:54pm, 10:01), (n, 10:19pm, 10:24pm), (n, 11:27pm, 11:34pm), (d, 10:03pm, 10:07pm)
    # (d, 10:25pm, 10:31pm), (f, 10:09pm, 10:11pm), (f, 10:45pm, 10:48pm), (a, 10:33pm, 10:42pm), (s, 10:53, 11:04pm)

    moods = [('s', '2023-4-20 9:08', '2023-4-20 9:10'), ('h', '2023-4-20 11:33', '2023-4-20 11:40'),
             ('h', '2023-4-20 23:46', '2023-4-21 00:09'),
             ('h', '2023-4-20 21:44', '2023-4-20 21:51'), ('h', '2023-4-20 23:08', '2023-4-20 23:24'),
             ('n', '2023-4-20 21:54', '2023-4-20 22:01'),
             ('n', '2023-4-20 22:19', '2023-4-20 22:24'), ('n', '2023-4-20 23:27', '2023-4-20 23:34'),
             ('d', '2023-4-20 22:03', '2023-4-20 22:07'),
             ('d', '2023-4-20 22:25', '2023-4-20 22:31'), ('f', '2023-4-20 22:09', '2023-4-20 22:11'),
             ('f', '2023-4-20 22:45', '2023-4-20 22:48'),
             ('a', '2023-4-20 22:33', '2023-4-20 22:42'), ('s', '2023-4-20 22:53', '2023-4-20 23:04')
             ]

    # convert strings to datetime objects and give time zone -0400 in tzinfo
    moods = [
        (
            x[0],
            datetime.datetime.strptime(x[1], '%Y-%m-%d %H:%M').replace(
                tzinfo=datetime.timezone(datetime.timedelta(hours=-4))),
            datetime.datetime.strptime(x[2], '%Y-%m-%d %H:%M').replace(
                tzinfo=datetime.timezone(datetime.timedelta(hours=-4)))
        )
        for x in moods
    ]

    # sort moods by start date, newest to oldest
    moods.sort(key=lambda x: x[1], reverse=True)
    return moods


def data_prep():
    raw_data = analysis.get_raw_data()
    mood_data = get_mood_data()

    # remove any raw data that is before the first mood data or after the last mood data
    rel_data = [x for x in raw_data if mood_data[-1][1] <= x.startDate <= mood_data[0][2]]

    print(f'Number of relevant data points: {len(rel_data)}')
    # Print number of data points for HR and BO2
    print(f'Number of HR data points: {len([x for x in rel_data if type(x) == HeartRate])}, Number of BO2 data points: '
          f'{len([x for x in rel_data if type(x) == BloodOxygen])}')

    # create data creator
    data_creator = DataCreator(rel_data, mood_data)

    tensor_ready_data = data_creator.get_tensor_ready_data()
    # save tensor_ready_data to file
    fn = f'..{os.sep}data{os.sep}tensor_ready_data.pkl'
    with open(fn, 'wb') as f:
        pickle.dump(tensor_ready_data, f)

    return tensor_ready_data


def does_data_exist():
    return True


if __name__ == '__main__':
    # print("Welcome to Sanji's Capstone Project!")
    # print("Do you want to run an XML and data analysis test? (y/n)")
    # if input().lower() == 'y':
    #     test_main()
    # print("Test Complete! \n ------------------ \n \n")
    #
    # print("Do you want to run the data preparation? (y/n)")
    # if input().lower() == 'y':
    #     data_prep()
    #
    # if does_data_exist():
    #     print("No data to use for NN. Exiting...")
    #     exit()
    #
    # print("Do you want to run the NN? (y/n)")
    # if input().lower() == 'y':
    #     # run NN
    #     pass
    trd = data_prep()
