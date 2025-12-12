from itertools import combinations

from linear_regression.test.constants import RFE_FEATURES, MUTUAL_FEATURES, ENSEMBLE_FEATURES, ANOVA_FEATURES, \
    CHI_FEATURES, RANDOM_FOREST_FEATURES

features_to_use = RFE_FEATURES
features_to_add = list(set(RANDOM_FOREST_FEATURES) - set(RFE_FEATURES))
# features_to_add = list(set(MUTUAL_FEATURES)  - set(features_to_use))
n = len(features_to_add)
print(n)

top_ten_models = {}
count = 0

combination_of_features = []
for r in range(1, n + 1):
    combination_of_features.extend(combinations(features_to_add, r))

print(len(combination_of_features))

# 8191 ensemble.union(anova)
# 8191 mutual
# 8191 forest
# 511 CHI


features = """checking_status_no checking
checking_status_<0
credit_history_critical/other existing credit
age_(-0.001, 25.0]
duration_(36.0, 60.0]
credit_amount_(10000.0, 20000.0]
checking_status_>=200
purpose_new car
other_parties_guarantor
duration_(24.0, 36.0]
savings_status_<100
purpose_retraining
age_(60.0, 80.0]
property_magnitude_real estate
savings_status_no known savings
installment_commitment
credit_amount_(2500.0, 5000.0]
employment_<1
property_magnitude_no known property
credit_history_no credits/all paid"""
l = set(features.split('\n'))
l2 = set(RFE_FEATURES)
print(l.difference(l2))
print(features.split('\n'))

