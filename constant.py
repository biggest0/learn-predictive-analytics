from enum import Enum


class PrintFrequency(Enum):
    ASCENDING = 'ascending'
    DESCENDING = 'descending'
    KEY = 'key'


TOP_FEATURES = ['accommodates', 'bedrooms', 'bathrooms', 'beds']
MAYBE_FEATURES = ['family_kid_friendly', 'dryer', 'indoor_fireplace', 'tv', 'washer']

TOP_20_FEATURE_FORWARD = ['accommodates', 'bedrooms', 'bathrooms', 'beds', 'family_kid_friendly', 'indoor_fireplace',
                          'tv', 'cable_tv', 'translation_missing:_en_hosting_amenity_49', 'dryer', 'washer',
                          'suitable_for_events', 'hot_tub', 'gym', 'pool', 'review_scores_rating', 'doorman',
                          'lock_on_bedroom_door', 'fire_extinguisher', 'free_parking_on_premises']
TOP_20_FEATURE_RECURSIVE = ['bathrooms', 'bedrooms', 'air_purifier', 'beachfront', 'crib', 'doorman', 'elevator',
                            'flat_smooth_pathway_to_front_door', 'free_parking_on_street', 'hand_or_paper_towel',
                            'indoor_fireplace', 'paid_parking_off_premises', 'path_to_entrance_lit_at_night',
                            'private_bathroom', 'smartlock', 'tv', 'table_corner_guards', 'toilet_paper',
                            'washer_dryer', 'wireless_internet']

TWO_TOP_20_FEATURE_FORWARD = ['accommodates', 'bedrooms', 'bathrooms', 'beds', 'room_type_Private room',
                              'family_kid_friendly', 'indoor_fireplace', 'tv', 'cable_tv',
                              'translation_missing:_en_hosting_amenity_49', 'dryer', 'washer', 'suitable_for_events',
                              'city_SF', 'room_type_Shared room', 'city_DC', 'cancellation_policy_strict', 'city_NYC',
                              'hot_tub', 'cancellation_policy_super_strict_60']
TWO_TOP_20_FEATURE_RECURSIVE = ['bathrooms', 'bedrooms', 'air_purifier', 'beachfront', 'doorman', 'elevator',
                                'free_parking_on_street', 'indoor_fireplace', 'paid_parking_off_premises',
                                'path_to_entrance_lit_at_night', 'roll_in_shower_with_chair', 'smartlock',
                                'suitable_for_events', 'city_Chicago', 'city_DC', 'city_SF', 'bed_type_Couch',
                                'room_type_Private room', 'room_type_Shared room',
                                'cancellation_policy_super_strict_60']

CATEGORICAL_FEATURES = ['property_type', 'room_type', 'bed_type', 'cancellation_policy', 'city',
                        'first_review', 'host_has_profile_pic', 'host_identity_verified',
                        'host_response_rate', 'host_since', 'instant_bookable', 'last_review',
                        'neighbourhood', 'zipcode']

COLUMNS_TO_DROP = ['neighbourhood', 'zipcode', 'translation_missing:_en_hosting_amenity_49']

COLUMNS_TO_DUMMY = ['city', 'bed_type', 'room_type', 'cancellation_policy', 'cleaning_fee']

COLUMNS_TO_BIN = ['host_since', ]
