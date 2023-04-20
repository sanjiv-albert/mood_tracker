import xml.etree.ElementTree as ET
from datetime import datetime

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


def extract_heart_rate_data(filename):
    tree = ET.parse("src/"+filename)
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
            heart_rate_obj = HeartRate(sourceName, sourceVersion, device, unit, startDate, endDate, value, motionContext)
            heart_rate_data.append(heart_rate_obj)
    heart_rate_data.sort(key=lambda x: x.endDate, reverse=True)
    return heart_rate_data
