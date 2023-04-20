import analysis


def main():
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




if __name__ == '__main__':
    main()
