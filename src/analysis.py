import os
import statistics as stats

import numpy as np

import XML_extraction
from constants import *

FILENAME = f'..{os.sep}data{os.sep}apple_health_export{os.sep}export.xml'


def get_raw_data():
    hr = XML_extraction.extract_heart_rate_data(FILENAME)
    bo2 = XML_extraction.extract_blood_O2_data(FILENAME)

    # Combine then sort by date with most recent first
    data = hr + bo2
    data.sort(key=lambda x: x.endDate, reverse=True)
    return data


def filter_data_by_source_name(data, source_name=SANJI_SOURCE_NAME):
    return [heart_rate_obj for heart_rate_obj in data if heart_rate_obj.sourceName == source_name]


def find_average_heartrate_by_motion_context(data):
    averages = {}
    for heart_rate_obj in data:
        if heart_rate_obj.motionContext in averages:
            averages[heart_rate_obj.motionContext].append(heart_rate_obj.value)
        else:
            averages[heart_rate_obj.motionContext] = [heart_rate_obj.value]
    return {key: np.mean(value) for key, value in averages.items()}, averages


def find_average_heartrate_by_time_of_day(data):
    averages = {}
    for heart_rate_obj in data:
        if heart_rate_obj.startDate.hour in averages:
            averages[heart_rate_obj.startDate.hour].append(heart_rate_obj.value)
        else:
            averages[heart_rate_obj.startDate.hour] = [heart_rate_obj.value]
    return {key: np.mean(value) for key, value in averages.items()}, averages


def mean_median_mode(data):
    data = np.array([x.value for x in data])
    return np.mean(data), np.median(data), stats.mode(data)


def get_sedentary_data(data):
    return [x for x in data if x.motionContext == MOTION_CONTEXTS['SEDENTARY']]
