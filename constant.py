from enum import Enum

class PrintFrequency(Enum):
    ASCENDING = 'ascending'
    DESCENDING = 'descending'
    KEY = 'key'


TOP_FEATURES = ['accommodates', 'bedrooms', 'bathrooms', 'beds']
MAYBE_FEATURES = ['family_kid_friendly', 'dryer', 'indoor_fireplace', 'tv', 'washer']