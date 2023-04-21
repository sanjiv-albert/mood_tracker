import contextlib
import pickle
from datetime import datetime

import xml.etree.ElementTree as ET


class HeartRate:
    def __init__(self, sourceName, sourceVersion, device, unit, startDate, endDate, value, motionContext):
        self.sourceName = sourceName
        self.sourceVersion = sourceVersion
        self.device = device
        self.unit = unit
        self.startDate = datetime.strptime(startDate, '%Y-%m-%d %H:%M:%S %z')
        self.endDate = datetime.strptime(endDate, '%Y-%m-%d %H:%M:%S %z')
        self.value = float(value)
        self.motionContext = int(motionContext) if motionContext is not None else None

    def __str__(self):
        return f"Source Name: {self.sourceName}\nStart Date: {self.startDate}\nEnd Date: {self.endDate}\nValue: {self.value}\nMotion Context: {self.motionContext}"


class BloodOxygen:
    def __init__(self, sourceName, sourceVersion, device, unit, startDate, endDate, value, motionContext):
        self.sourceName = sourceName
        self.sourceVersion = sourceVersion
        self.device = device
        self.unit = unit
        self.startDate = datetime.strptime(startDate, '%Y-%m-%d %H:%M:%S %z')
        self.endDate = datetime.strptime(endDate, '%Y-%m-%d %H:%M:%S %z')
        self.value = float(value)
        self.motionContext = int(motionContext) if motionContext is not None else None

    def __str__(self):
        return f"Source Name: {self.sourceName}\nStart Date: {self.startDate}\nEnd Date: {self.endDate}\nValue: {self.value}\nMotion Context: {self.motionContext}"


def extract_heart_rate_data(filename):
    pck_file = filename.replace('.xml', '_HR_.pck')
    with contextlib.suppress(FileNotFoundError):
        with open('', 'rb') as f:
            return pickle.load(f)

    tree = ET.parse(filename)
    root = tree.getroot()
    heart_rate_data = []
    for record in root.iter('Record'):
        if record.get('type') == 'HKQuantityTypeIdentifierHeartRate':
            sourceName = record.get('sourceName')
            sourceVersion = record.get('sourceVersion')
            device = record.get('device')
            unit = record.get('unit')
            startDate = record.get('startDate')
            endDate = record.get('endDate')
            value = float(record.get('value'))
            metadata = record.find('MetadataEntry')
            motionContext = metadata.get('value') if metadata is not None else None
            heart_rate_obj = HeartRate(sourceName, sourceVersion, device, unit, startDate, endDate, value,
                                       motionContext)
            heart_rate_data.append(heart_rate_obj)
    heart_rate_data.sort(key=lambda x: x.endDate, reverse=True)
    with open(pck_file, 'wb') as f:
        pickle.dump(heart_rate_data, f)
    return heart_rate_data


def extract_blood_O2_data(filename):
    pck_file = filename.replace('.xml', '_O2_.pck')
    with contextlib.suppress(FileNotFoundError):
        with open(pck_file, 'rb') as f:
            return pickle.load(f)

    tree = ET.parse(filename)
    root = tree.getroot()
    blood_O2_data = []
    for record in root.iter('Record'):
        if record.get('type') == 'HKQuantityTypeIdentifierOxygenSaturation':
            sourceName = record.get('sourceName')
            sourceVersion = record.get('sourceVersion')
            device = record.get('device')
            unit = record.get('unit')
            startDate = record.get('startDate')
            endDate = record.get('endDate')
            value = float(record.get('value'))
            metadata = record.find('MetadataEntry')
            motionContext = None
            blood_O2_obj = BloodOxygen(sourceName, sourceVersion, device, unit, startDate, endDate, value,
                                       motionContext)
            blood_O2_data.append(blood_O2_obj)
    blood_O2_data.sort(key=lambda x: x.endDate, reverse=True)
    with open(pck_file, 'wb') as f:
        pickle.dump(blood_O2_data, f)
    return blood_O2_data
