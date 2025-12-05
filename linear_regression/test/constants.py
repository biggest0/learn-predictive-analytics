NUMERIC_COLS_MISSING_DATA = ['credit_amount']  # 713/750
CATEGORICAL_COLS_MISSING_DATA = ['checking_status']  # 710/750

CATEGORICAL_COLS_TO_DUMMY = ['checking_status', 'credit_history', 'purpose', 'savings_status', 'employment',
                             'personal_status', 'other_parties', 'property_magnitude', 'other_payment_plans', 'housing',
                             'job', 'own_telephone', 'foreign_worker']

COLS_TO_BIN = {
    "credit_amount": [0, 1500, 2500, 5000, 10000, 20000],
    "age": [0, 25, 35, 45, 60, 80],
    "duration": [0, 12, 18, 24, 36, 60],
}

ANOVA_FEATURES = ['checking_status_no checking', 'checking_status_<0', 'credit_history_critical/other existing credit',
                  'duration', 'housing_own', 'credit_amount', 'credit_history_no credits/all paid',
                  'age_(-0.001, 25.0]', 'savings_status_<100', 'duration_(36.0, 60.0]',
                  'property_magnitude_no known property', 'age', 'property_magnitude_real estate', 'purpose_radio/tv',
                  'credit_amount_(10000.0, 20000.0]', 'purpose_new car', 'duration_(-0.001, 12.0]', 'housing_rent',
                  'duration_(24.0, 36.0]', 'savings_status_no known savings']

MUTUAL_FEATURES = ['checking_status_<0', 'checking_status_no checking', 'duration_(36.0, 60.0]', 'duration',
                   'num_dependents', 'installment_commitment', 'employment_<1', 'purpose_domestic appliance',
                   'age_(25.0, 35.0]', 'credit_amount_(10000.0, 20000.0]',
                   'credit_history_critical/other existing credit', 'purpose_furniture/equipment',
                   'credit_amount_(2500.0, 5000.0]', 'savings_status_<100', 'job_unskilled resident', 'credit_amount',
                   'job_unemp/unskilled non res', 'credit_history_delayed previously', 'purpose_other',
                   'credit_amount_(5000.0, 10000.0]']

RANDOM_FOREST_FEATURES = ['credit_amount', 'age', 'duration', 'checking_status_no checking', 'checking_status_<0',
                          'installment_commitment', 'residence_since', 'credit_history_critical/other existing credit',
                          'existing_credits', 'savings_status_<100', 'purpose_new car', 'housing_own',
                          'other_payment_plans_none', 'job_skilled', 'own_telephone_yes', 'personal_status_male single',
                          'purpose_radio/tv', 'property_magnitude_real estate', 'employment_<1',
                          'credit_amount_(2500.0, 5000.0]']

ENSEMBLE_FEATURES = ['checking_status_<0', 'checking_status_no checking',
                     'credit_history_critical/other existing credit', 'savings_status_<100', 'purpose_new car',
                     'housing_own', 'credit_amount_(10000.0, 20000.0]', 'credit_amount', 'duration',
                     'duration_(36.0, 60.0]', 'property_magnitude_real estate', 'age_(-0.001, 25.0]',
                     'duration_(24.0, 36.0]', 'age', 'purpose_radio/tv', 'installment_commitment', 'employment_<1',
                     'property_magnitude_no known property', 'credit_amount_(2500.0, 5000.0]',
                     'credit_history_delayed previously']

RFE_FEATURES = ['checking_status_no checking', 'checking_status_<0', 'credit_history_critical/other existing credit',
                'age_(-0.001, 25.0]', 'duration_(36.0, 60.0]', 'credit_amount_(10000.0, 20000.0]',
                'checking_status_>=200', 'purpose_new car',
                'other_parties_guarantor',
                'duration_(24.0, 36.0]', 'savings_status_<100', 'purpose_retraining',
                'age_(60.0, 80.0]', 'credit_history_delayed previously', 'other_payment_plans_none',
                'purpose_used car', 'housing_own', 'foreign_worker_yes',
                'credit_history_existing paid', 'purpose_education',
                ]

CHI_FEATURES = ['purpose_new car', 'checking_status_no checking', 'purpose_radio/tv', 'duration_(24.0, 36.0]',
                'savings_status_<100', 'age_(-0.001, 25.0]', 'property_magnitude_real estate', 'age',
                'duration_(36.0, 60.0]', 'housing_own', 'credit_amount', 'duration', 'checking_status_<0',
                'credit_history_no credits/all paid', 'purpose_education', 'property_magnitude_no known property',
                'credit_amount_(10000.0, 20000.0]', 'housing_rent', 'duration_(-0.001, 12.0]',
                'credit_history_critical/other existing credit']

RFE_CHI_UNION = ['credit_amount', 'purpose_education', 'purpose_new car', 'housing_rent', 'age',
                 'duration_(24.0, 36.0]', 'age_(-0.001, 25.0]', 'checking_status_<0', 'duration_(-0.001, 12.0]',
                 'other_payment_plans_none', 'credit_history_existing paid', 'age_(60.0, 80.0]', 'purpose_radio/tv',
                 'housing_own', 'purpose_used car', 'credit_history_delayed previously', 'checking_status_>=200',
                 'checking_status_no checking', 'purpose_retraining', 'property_magnitude_real estate',
                 'credit_history_critical/other existing credit', 'duration_(36.0, 60.0]', 'other_parties_guarantor',
                 'foreign_worker_yes', 'credit_amount_(10000.0, 20000.0]', 'property_magnitude_no known property',
                 'savings_status_<100', 'credit_history_no credits/all paid', 'duration']

RFE_CHI_INTERSECT = ['purpose_education', 'purpose_new car', 'credit_amount_(10000.0, 20000.0]',
                     'duration_(24.0, 36.0]', 'age_(-0.001, 25.0]', 'checking_status_no checking', 'checking_status_<0',
                     'savings_status_<100', 'housing_own', 'credit_history_critical/other existing credit',
                     'duration_(36.0, 60.0]']

TOP_20_FEATURES = [
    # Tier 1: Consensus features (selected by 5 methods)
    'checking_status_<0',
    'checking_status_no checking',
    'credit_history_critical/other existing credit',
    'savings_status_<100',

    # Tier 2: Strong consensus (selected by 4 methods)
    'purpose_new car',
    'housing_own',
    'credit_amount_(10000.0, 20000.0]',
    'credit_amount',
    'duration',
    'duration_(36.0, 60.0]',

    # Tier 3: Important features from multiple methods (3 methods)
    'property_magnitude_real estate',
    'age_(-0.001, 25.0]',
    'duration_(24.0, 36.0]',
    'age',
    'purpose_radio/tv',

    # Tier 4: High importance in specific methods + domain relevance
    'installment_commitment',  # MI + RF (2 methods) + important for credit risk
    'employment_<1',  # MI + RF (2 methods) + employment stability matters
    'checking_status_>=200',  # RFE selected + checking account is crucial
    'other_payment_plans_none',  # RFE + RF (2 methods)
    'job_skilled',  # RF selected + employment quality matters
]

CANDIDATE_FEATURES = [
    # High consensus but similar to included features
    'credit_amount_(2500.0, 5000.0]',  # MI + RF - another credit amount bin

    # Selected by multiple methods
    'credit_history_delayed previously',  # MI + RFE (2 methods)
    'purpose_education',  # Chi2 + RFE (2 methods)
    'property_magnitude_no known property',  # ANOVA + Chi2 (2 methods)
    'duration_(-0.001, 12.0]',  # ANOVA + Chi2 (2 methods)

    # High importance in Random Forest
    'residence_since',  # RF ranked #7
    'existing_credits',  # RF ranked #9
    'own_telephone_yes',  # RF ranked #15
    'personal_status_male single',  # RF ranked #16

    # High MI score (captures non-linear relationships)
    'num_dependents',  # MI ranked #5
]

FEATURES_TEST_1 = ['checking_status_>=200', ]

ANOVA_ACTUAL = ['checking_status_no checking', 'checking_status_<0', 'credit_history_critical/other existing credit',
                'housing_own', 'credit_history_no credits/all paid', 'age_(-0.001, 25.0]', 'savings_status_<100',
                'duration_(36.0, 60.0]', 'property_magnitude_no known property', 'property_magnitude_real estate',
                'purpose_radio/tv', 'credit_amount_(10000.0, 20000.0]', 'purpose_new car', 'duration_(-0.001, 12.0]',
                'housing_rent', 'duration_(24.0, 36.0]', 'savings_status_no known savings', 'employment_<1',
                'other_payment_plans_none', 'personal_status_male single']

MUTUAL = ['checking_status_no checking', 'checking_status_<0', 'duration_(24.0, 36.0]', 'purpose_furniture/equipment',
          'purpose_repairs', 'employment_unemployed', 'employment_4<=X<7', 'employment_>=7',
          'credit_history_critical/other existing credit', 'purpose_radio/tv', 'purpose_education', 'employment_<1',
          'purpose_other', 'credit_amount_(1500.0, 2500.0]', 'personal_status_male div/sep', 'age_(-0.001, 25.0]',
          'job_skilled', 'credit_amount_(10000.0, 20000.0]', 'purpose_domestic appliance', 'housing_own']

FOREST = ['installment_commitment', 'checking_status_no checking', 'checking_status_<0', 'residence_since',
          'credit_history_critical/other existing credit', 'savings_status_<100', 'own_telephone_yes',
          'existing_credits', 'age_(-0.001, 25.0]', 'purpose_new car', 'personal_status_male single', 'job_skilled',
          'duration_(-0.001, 12.0]', 'employment_<1', 'property_magnitude_life insurance', 'housing_own',
          'property_magnitude_real estate', 'other_payment_plans_none', 'employment_>=7', 'purpose_radio/tv']

ENSEMBLE = ['checking_status_<0', 'age_(-0.001, 25.0]', 'housing_own', 'checking_status_no checking',
            'credit_history_critical/other existing credit', 'purpose_radio/tv', 'purpose_new car',
            'savings_status_<100', 'employment_<1', 'duration_(24.0, 36.0]', 'credit_amount_(10000.0, 20000.0]',
            'purpose_education', 'other_payment_plans_none', 'duration_(-0.001, 12.0]', 'duration_(36.0, 60.0]',
            'property_magnitude_real estate', 'credit_history_no credits/all paid',
            'property_magnitude_no known property', 'housing_rent', 'personal_status_male single']

RFE = ['checking_status_<0', 'checking_status_>=200', 'savings_status_<100', 'checking_status_no checking',
       'purpose_used car', 'duration_(24.0, 36.0]', 'other_payment_plans_none', 'purpose_education',
       'duration_(36.0, 60.0]', 'other_parties_guarantor', 'age_(60.0, 80.0]',
       'credit_history_critical/other existing credit', 'credit_amount_(10000.0, 20000.0]',
       'credit_history_existing paid', 'purpose_new car', 'housing_own', 'purpose_retraining',
       'credit_history_delayed previously', 'foreign_worker_yes', 'age_(-0.001, 25.0]']

CHI = ['savings_status_no known savings', 'purpose_radio/tv', 'checking_status_<0', 'property_magnitude_real estate',
       'property_magnitude_no known property', 'savings_status_<100', 'checking_status_no checking',
       'duration_(24.0, 36.0]', 'housing_rent', 'credit_history_no credits/all paid', 'age_(-0.001, 25.0]',
       'duration_(-0.001, 12.0]', 'purpose_education', 'duration_(36.0, 60.0]', 'savings_status_>=1000',
       'credit_history_critical/other existing credit', 'credit_amount_(10000.0, 20000.0]', 'purpose_new car',
       'housing_own', 'employment_<1']

FFS = ['checking_status_no checking', 'checking_status_<0', 'credit_history_critical/other existing credit',
       'housing_own', 'credit_history_no credits/all paid', 'age_(-0.001, 25.0]', 'savings_status_<100',
       'duration_(36.0, 60.0]', 'property_magnitude_no known property', 'property_magnitude_real estate',
       'purpose_radio/tv', 'credit_amount_(10000.0, 20000.0]', 'purpose_new car', 'duration_(-0.001, 12.0]',
       'housing_rent', 'duration_(24.0, 36.0]', 'savings_status_no known savings', 'employment_<1',
       'other_payment_plans_none', 'personal_status_male single']



SELECTED_FEATURES = ['checking_status_no checking', 'checking_status_<0',
                     'credit_history_critical/other existing credit',
                     'age_(-0.001, 25.0]', 'duration_(36.0, 60.0]', 'credit_amount_(10000.0, 20000.0]',
                     'checking_status_>=200', 'purpose_new car', 'other_parties_guarantor', 'duration_(24.0, 36.0]',
                     'savings_status_<100', 'purpose_retraining', 'age_(60.0, 80.0]', 'property_magnitude_real estate',
                     'savings_status_no known savings', 'installment_commitment', 'credit_amount_(2500.0, 5000.0]',
                     'employment_<1', 'property_magnitude_no known property', 'credit_history_no credits/all paid']


NEW_BEST_20_FEATURES = [
    # Tier 1: Appears in 6-7 methods (Strongest consensus)
    'checking_status_<0',  # 7/7 methods
    'checking_status_no checking',  # 7/7 methods
    'credit_history_critical/other existing credit',  # 7/7 methods
    'age_(-0.001, 25.0]',  # 7/7 methods
    'savings_status_<100',  # 6/7 methods
    'housing_own',  # 7/7 methods
    'purpose_new car',  # 6/7 methods
    'purpose_radio/tv',  # 6/7 methods
    'employment_<1',  # 6/7 methods

    # Tier 2: Appears in 5 methods (Strong consensus)
    'duration_(24.0, 36.0]',  # 5/7 methods
    'credit_amount_(10000.0, 20000.0]',  # 5/7 methods
    'duration_(36.0, 60.0]',  # 5/7 methods
    'duration_(-0.001, 12.0]',  # 5/7 methods
    'property_magnitude_real estate',  # 5/7 methods
    'other_payment_plans_none',  # 5/7 methods

    # Tier 3: Appears in 4 methods (Good consensus)
    'purpose_education',  # 4/7 methods
    'property_magnitude_no known property',  # 4/7 methods
    'personal_status_male single',  # 4/7 methods
    'credit_history_no credits/all paid',  # 4/7 methods

    # Tier 4: High importance in specific methods + domain relevance
    'installment_commitment',  # Random Forest #1, domain critical
]
