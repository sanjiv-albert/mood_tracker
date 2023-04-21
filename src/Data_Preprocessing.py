# @Author: Sanji Albert (sanjiv.albert.19@cnu.edu)
# Description: This is the neural network that that will be trained and eventually
#              used to predict mood based on the HR and Blood O2 data from the apple watch


import numpy as np

from XML_extraction import HeartRate, BloodOxygen


def get_obj_time(obj_time, times):
    # obj_time is a datetime object
    # times is an array of tuples of (mood, datetime(start), datetime(end))
    for time in times:
        if time[1] <= obj_time <= time[2]:
            return time
    return None


def convert_to_array(data, times):
    # data is a set of heart rate and blood oxygen objects
    # we want to convert this to an array
    # Format will be
    # Time
    # Mood
    # Type: HR or Blood O2
    # Value
    # Motion Context
    new_data = []

    for o in data:
        o.startDate = int(o.startDate.timestamp())
        o.endDate = int(o.endDate.timestamp())

    for i in range(len(data)):
        object = data[i]
        obj_time = int(object.startDate + (object.endDate - object.startDate) / 2)
        related_time = get_obj_time(obj_time, times)
        if related_time is not None:
            if type(object) is HeartRate:
                new_data.append([obj_time, related_time[0], 'HR', object.value, object.motionContext])
            elif type(object) is BloodOxygen:
                new_data.append([obj_time, related_time[0], 'O2', object.value, object.motionContext])
            else:
                raise TypeError('Object is not a heart rate or blood oxygen object. Type is ' + str(type(object)))
    return new_data


def create_second_by_second_data(data, mood_times):
    hr_second_by_second = []
    o2_second_by_second = []
    # start by filling in existing data, the format will be [time in epoch seconds, value]
    for i in range(len(data)):
        if data[i][2] == 'HR':
            hr_second_by_second.append([data[i][0], data[i][3]])
        else:
            o2_second_by_second.append([data[i][0], data[i][3]])

    # Sort the data to be chronological, this is important for the interpolation. Oldest data first
    hr_second_by_second.sort(key=lambda x: x[0])
    o2_second_by_second.sort(key=lambda x: x[0])

    # now fill in the missing data, fill in the missing data using polynomial interpolation within 1 minute. Only fill in
    # data that is missing for less than 1 minute, any longer than that shows a break between data collection times
    # and we don't want to fill in that data

    # first fill in the HR data
    i, cursor = 0, 0
    while i < len(hr_second_by_second) or cursor >= len(hr_second_by_second) - 1:
        current = hr_second_by_second[i]
        prev = hr_second_by_second[i - 1] if i > 0 else current
        next = hr_second_by_second[i + 1] if i + 1 < len(hr_second_by_second) else current

        # check if next data point is more than 1 minute away or if we are at the end of the data
        if next[0] - current[0] > 60 or i == len(hr_second_by_second) - 1:
            i += 1
        elif next[0] - current[0] == 1:
            # check if we already have data for the next second
            i += 1
        else:
            # fill in the missing data
            # go through a 2 minute by 2 minute basis and calculate curve fit for each minute and fill in the missing data
            # for that minute
            current_time = current[0]
            two_min_subset = [current]
            cursor = i + 1
            while cursor < len(hr_second_by_second) and hr_second_by_second[cursor][0] - current_time < 120:
                two_min_subset.append(hr_second_by_second[cursor])
                cursor += 1

            # Now we calculate the curve fit to 3rd degree polynomial for the 2 minute subset
            sub = np.array(two_min_subset)
            x = sub[:, 0]
            y = sub[:, 1]
            z = np.polyfit(x, y, 3)
            f = np.poly1d(z)
            sub_max = max(sub[:, 1])
            sub_min = min(sub[:, 1])

            add_index = hr_second_by_second.index(two_min_subset[0]) + 1
            # Now we fill in the missing data
            for adder in range(two_min_subset[0][0] + 1, two_min_subset[-1][0]):
                if adder not in sub[:, 0]:
                    # interpolated value must be between the min and max of the subset
                    interpolated_value = f(adder) if sub_min <= f(adder) <= sub_max else sub_min if f(
                        adder) < sub_min else sub_max

                    hr_second_by_second.insert(i + add_index, [adder, interpolated_value])
                    add_index += 1

            hr_second_by_second.sort(key=lambda x: x[0])
            sub_end = two_min_subset[-1][0]
            i += 1
            # move i to the end of the subset + 1
            while i < len(hr_second_by_second) and hr_second_by_second[i][0] <= sub_end:
                i += 1

    # now fill in the O2 data
    hr_second_by_second.sort(key=lambda x: x[0])
    original_o2 = o2_second_by_second.copy()
    for (i, _) in hr_second_by_second:
        if i not in [x[0] for x in o2_second_by_second]:
            # if blood oxygen measurement is within 5 min of i, then use that measurement, otherwise use avg of previous and next

            # find the closest measurement before i
            closest_before = None
            for j in range(len(original_o2)):
                if original_o2[j][0] < i:
                    closest_before = original_o2[j]
                else:
                    break
            # find the closest measurement after i
            closest_after = None
            for j in range(len(original_o2) - 1, -1, -1):
                if original_o2[j][0] > i:
                    closest_after = original_o2[j]
                else:
                    break
            # if both are found, then use polynomial interpolation to fill in the missing data
            if closest_before is not None and closest_after is not None:
                x = np.array([closest_before[0], closest_after[0]])
                y = np.array([closest_before[1], closest_after[1]])
                z = np.polyfit(x, y, 1)
                f = np.poly1d(z)
                o2_second_by_second.append([i, f(i)])
            # if only one is found, then use that measurement
            elif closest_before is not None:
                o2_second_by_second.append([i, closest_before[1]])
            elif closest_after is not None:
                o2_second_by_second.append([i, closest_after[1]])
            # if neither are found, then use 99.5 as the measurement
            else:
                o2_second_by_second.append([i, 99.5])

    # Now recombine the HR and O2 data
    # Format of the data is [[mood, time, hr, o2], ...]
    combined_data = []
    for i in range(len(hr_second_by_second)):
        append = True
        time = hr_second_by_second[i][0]
        hr = hr_second_by_second[i][1]
        # find the o2 data for the same time, if not find closest time
        o2 = None
        for j in range(len(o2_second_by_second)):
            if o2_second_by_second[j][0] == time:
                o2 = o2_second_by_second[j][1]
                break
            elif o2_second_by_second[j][0] > time:
                o2 = o2_second_by_second[j - 1][1]
                break
        if o2 is None:
            if i < len(hr_second_by_second) / 2:
                o2 = o2_second_by_second[0][1]
            else:
                o2 = o2_second_by_second[-1][1]
        mood = None
        for j in range(len(mood_times)):
            if mood_times[j][1] <= time <= mood_times[j][2]:
                mood = mood_times[j][0]
                break
        if mood is None:
            if 0 <= i < len(hr_second_by_second):
                mood = combined_data[i - 1][0]
            else:
                append = False
        if append:
            combined_data.append([str(mood), int(time), float(hr), float(o2)])
    combined_data.sort(key=lambda x: x[1])
    return combined_data


def get_hr_10s_avg(time, np_full_data):
    # 5 seconds before and 5 seconds after
    subset = np_full_data[np.where((np_full_data[:, 1] >= time - 5) & (np_full_data[:, 1] <= time + 5))]
    return np.mean(subset[:, 2])


def get_hr_1m_avg(time, np_full_data):
    # 30 seconds before and 30 seconds after
    subset = np_full_data[np.where((np_full_data[:, 1] >= time - 30) & (np_full_data[:, 1] <= time + 30))]
    return np.mean(subset[:, 2])


def get_hr_min_1m(time, np_full_data):
    # 30 seconds before and 30 seconds after
    subset = np_full_data[np.where((np_full_data[:, 1] >= time - 30) & (np_full_data[:, 1] <= time + 30))]
    return np.min(subset[:, 2])


def get_hr_max_1m(time, np_full_data):
    # 30 seconds before and 30 seconds after
    subset = np_full_data[np.where((np_full_data[:, 1] >= time - 30) & (np_full_data[:, 1] <= time + 30))]
    return np.max(subset[:, 2])


def get_hr_std_dev_short(time, np_full_data):
    # 5 seconds before and 5 seconds after
    subset = np_full_data[np.where((np_full_data[:, 1] >= time - 5) & (np_full_data[:, 1] <= time + 5))]
    return np.std(subset[:, 2])


def get_hr_std_dev_long(time, np_full_data):
    # 30 seconds before and 30 seconds after
    subset = np_full_data[np.where((np_full_data[:, 1] >= time - 30) & (np_full_data[:, 1] <= time + 30))]
    return np.std(subset[:, 2])


def get_hr_1m_entropy(time, np_full_data):
    # 30 seconds before and 30 seconds after
    subset = np_full_data[np.where((np_full_data[:, 1] >= time - 30) & (np_full_data[:, 1] <= time + 30))]
    mean = np.mean(subset[:, 2])
    entropy = 0
    for i in range(len(subset)):
        entropy += (subset[i, 2] - mean) ** 2
    return entropy


def get_hr_10s_entropy(time, np_full_data):
    # 5 seconds before and 5 seconds after
    subset = np_full_data[np.where((np_full_data[:, 1] >= time - 5) & (np_full_data[:, 1] <= time + 5))]
    mean = np.mean(subset[:, 2])
    entropy = 0
    for i in range(len(subset)):
        entropy += (subset[i, 2] - mean) ** 2
    return entropy


def get_hr_slope(time, np_full_data):
    # 5 seconds before and 5 seconds after
    subset = np_full_data[np.where((np_full_data[:, 1] >= time - 5) & (np_full_data[:, 1] <= time + 5))]
    if len(subset) < 2:
        return 0
    inst_slope = np.polyfit(subset[:, 1], subset[:, 2], 1)
    return inst_slope[0]


def get_o2_10s_avg(time, np_full_data):
    # 5 seconds before and 5 seconds after
    subset = np_full_data[np.where((np_full_data[:, 1] >= time - 5) & (np_full_data[:, 1] <= time + 5))]
    return np.mean(subset[:, 3])


def get_o2_1m_avg(time, np_full_data):
    # 30 seconds before and 30 seconds after
    subset = np_full_data[np.where((np_full_data[:, 1] >= time - 30) & (np_full_data[:, 1] <= time + 30))]
    return np.mean(subset[:, 3])


def get_o2_min_1m(time, np_full_data):
    # 30 seconds before and 30 seconds after
    subset = np_full_data[np.where((np_full_data[:, 1] >= time - 30) & (np_full_data[:, 1] <= time + 30))]
    return np.min(subset[:, 3])


def get_o2_max_1m(time, np_full_data):
    # 30 seconds before and 30 seconds after
    subset = np_full_data[np.where((np_full_data[:, 1] >= time - 30) & (np_full_data[:, 1] <= time + 30))]
    return np.max(subset[:, 3])


def get_o2_std_dev_short(time, np_full_data):
    # 5 seconds before and 5 seconds after
    subset = np_full_data[np.where((np_full_data[:, 1] >= time - 5) & (np_full_data[:, 1] <= time + 5))]
    return np.std(subset[:, 3])


def get_o2_std_dev_long(time, np_full_data):
    # 30 seconds before and 30 seconds after
    subset = np_full_data[np.where((np_full_data[:, 1] >= time - 30) & (np_full_data[:, 1] <= time + 30))]
    return np.std(subset[:, 3])


def get_o2_1m_entropy(time, np_full_data):
    # 30 seconds before and 30 seconds after
    subset = np_full_data[np.where((np_full_data[:, 1] >= time - 30) & (np_full_data[:, 1] <= time + 30))]
    mean = np.mean(subset[:, 3])
    entropy = 0
    for i in range(len(subset)):
        entropy += (subset[i, 3] - mean) ** 2
    return entropy


def get_o2_10s_entropy(time, np_full_data):
    # 5 seconds before and 5 seconds after
    subset = np_full_data[np.where((np_full_data[:, 1] >= time - 5) & (np_full_data[:, 1] <= time + 5))]
    mean = np.mean(subset[:, 3])
    entropy = 0
    for i in range(len(subset)):
        entropy += (subset[i, 3] - mean) ** 2
    return entropy


def get_o2_slope(time, np_full_data):
    # 5 seconds before and 5 seconds after
    subset = np_full_data[np.where((np_full_data[:, 1] >= time - 5) & (np_full_data[:, 1] <= time + 5))]
    if len(subset) < 2:
        return 0
    inst_slope = np.polyfit(subset[:, 1], subset[:, 3], 1)
    return inst_slope[0]


def convert_to_mood_array(mood_times):
    # change mood times to an array of tuples of (mood, datetime(start), datetime(end))
    return [(m, int(s.timestamp()), int(e.timestamp())) for m, s, e in mood_times]


class DataCreator:

    def __init__(self, data, mood_times):
        self.mood_times = convert_to_mood_array(mood_times)
        self.data = convert_to_array(data, self.mood_times)

        # times is an array of tuples of (mood, datetime(start), datetime(end))

    def get_tensor_ready_data(self):
        # This will return the data in the format that the neural network will want it
        # The format will be two arrays, one for the input and one for the output
        # The input will be a 2D array with the following format:
        # [Time, Motion Context,
        # HR, 10s Avg HR, 1m Avg HR, min 1m HR, max 1m HR, std_dev hr short term, std_dev hr long term, 1m Entropy HR, 10s Entropy HR, HR Slope
        # O2, 10s Avg O2, 1m Avg O2, min 1m O2, max 1m O2, std_dev o2 short term, std_dev o2 long term, 1m Entropy O2, 10s Entropy O2, O2 Slope]
        # The output will be a 1D array with the following format:
        # [Mood]
        # Any missing data will be calculated using interpolation

        full_data = create_second_by_second_data(self.data, self.mood_times)
        # [ [mood, time, hr, o2], ...]
        np_full_data = np.array([[0, t, h, o] for m, t, h, o in full_data])
        # Now we need to create the input and output arrays
        in_arr = []
        out_arr = []

        # We will go through the data and create the input and output arrays, motion context will be
        # 0 for now, can be added later
        for i in range(len(np_full_data)):
            time = np_full_data[i][1]
            mc = 0

            hr = np_full_data[i][2]
            hr_10s_avg = get_hr_10s_avg(time, np_full_data)
            hr_1m_avg = get_hr_1m_avg(time, np_full_data)
            hr_min_1m = get_hr_min_1m(time, np_full_data)
            hr_max_1m = get_hr_max_1m(time, np_full_data)
            hr_std_dev_short = get_hr_std_dev_short(time, np_full_data)
            hr_std_dev_long = get_hr_std_dev_long(time, np_full_data)
            hr_1m_entropy = get_hr_1m_entropy(time, np_full_data)
            hr_10s_entropy = get_hr_10s_entropy(time, np_full_data)
            hr_slope = get_hr_slope(time, np_full_data)

            o2 = np_full_data[i][3]
            o2_10s_avg = get_o2_10s_avg(time, np_full_data)
            o2_1m_avg = get_o2_1m_avg(time, np_full_data)
            o2_min_1m = get_o2_min_1m(time, np_full_data)
            o2_max_1m = get_o2_max_1m(time, np_full_data)
            o2_std_dev_short = get_o2_std_dev_short(time, np_full_data)
            o2_std_dev_long = get_o2_std_dev_long(time, np_full_data)
            o2_1m_entropy = get_o2_1m_entropy(time, np_full_data)
            o2_10s_entropy = get_o2_10s_entropy(time, np_full_data)
            o2_slope = get_o2_slope(time, np_full_data)

            in_arr.append([time, mc, hr, hr_10s_avg, hr_1m_avg, hr_min_1m, hr_max_1m, hr_std_dev_short, hr_std_dev_long,
                           hr_1m_entropy, hr_10s_entropy, hr_slope, o2, o2_10s_avg, o2_1m_avg, o2_min_1m, o2_max_1m,
                           o2_std_dev_short, o2_std_dev_long, o2_1m_entropy, o2_10s_entropy, o2_slope])
            # The output will be the mood
            # mood options are (happiness, sadness, anger, fear, disgust, neutral)
            # we will use the following format:
            # [happiness, sadness, anger, fear, disgust, neutral] floats 0 to 255

            mood = full_data[i][0]
            if mood == 'h':
                out_arr.append([255, 0, 0, 0, 0, 0])
            elif mood == 's':
                out_arr.append([0, 255, 0, 0, 0, 0])
            elif mood == 'a':
                out_arr.append([0, 0, 255, 0, 0, 0])
            elif mood == 'f':
                out_arr.append([0, 0, 0, 255, 0, 0])
            elif mood == 'd':
                out_arr.append([0, 0, 0, 0, 255, 0])
            elif mood == 'n':
                out_arr.append([0, 0, 0, 0, 0, 255])
            else:
                print("Error: mood not recognized")
                out_arr.append([0, 0, 0, 0, 0, 0])
        return in_arr, out_arr
