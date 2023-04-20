import XML_extraction
from constants import *
import numpy as np
import statistics as stats

FILENAME = 'export.xml'


def get_raw_data():
    return XML_extraction.extract_heart_rate_data(FILENAME)


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
