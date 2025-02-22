import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from causalinference import CausalModel 
import markdown
import pandas as pd
import numpy as np


HCRIS = pd.read_csv('submission3/data/output/HCRIS_Data.csv')
#from data_summary_v3 import HCRIS_data_filtered as HCRIS_data

hcris_2012 = HCRIS[HCRIS['year'] == 2012]

#Calculating estimated price for 2012
hcris_2012['discount_factor'] = 1 - hcris_2012['tot_discounts'] / hcris_2012['tot_charges']
hcris_2012['price_num'] = (
    (hcris_2012['ip_charges'] + hcris_2012['icu_charges'] + hcris_2012['ancillary_charges'])
    * hcris_2012['discount_factor'] - hcris_2012['tot_mcare_payment'])
hcris_2012['price_denom'] = hcris_2012['tot_discharges'] - hcris_2012['mcare_discharges']
hcris_2012['price'] = hcris_2012['price_num'] / hcris_2012['price_denom']

# Cleaning the  data
hcris_2012 = hcris_2012[(hcris_2012['price_denom'] > 100) & (hcris_2012['price_num'] > 0) & (hcris_2012['price'] > 0)]
hcris_2012 = hcris_2012[hcris_2012['beds'] > 30]
hcris_2012 = hcris_2012[hcris_2012['price'] < 100000]  

#NA payments
hcris_2012['hvbp_payment'] = hcris_2012['hvbp_payment'].fillna(0)
hcris_2012['hrrp_payment'] = hcris_2012['hrrp_payment'].fillna(0).abs()

#Defining penalty 
hcris_2012['penalty'] = (hcris_2012['hvbp_payment'] + hcris_2012['hrrp_payment'] < 0).astype(int)


# Calculate average price for penalized vs non-penalized hospitals
mean_penalized = round(hcris_2012.loc[hcris_2012['penalty'] == 1, 'price'].mean(), 2)
mean_non_penalized = round(hcris_2012.loc[hcris_2012['penalty'] == 0, 'price'].mean(), 2)

print(f"Average price for penalized hospitals in 2012: {mean_penalized}")
print(f"Average price for non-penalized hospitals in 2012: {mean_non_penalized}")

# Q6
hcris_2012['beds_quartile'] = pd.qcut(hcris_2012['beds'], 4, labels=[1, 2, 3, 4])

# Create indicator variables for each quartile
for i in range(1, 5):
    hcris_2012[f'quartile_{i}'] = (hcris_2012['beds_quartile'] == i).astype(int)


# Calculate average price for treated and control groups within each quartile
Avg_per_group = []
for i in range(1, 5):
    treated_mean = hcris_2012.loc[(hcris_2012[f'quartile_{i}'] == 1) & (hcris_2012['penalty'] == 1), 'price'].mean()
    control_mean = hcris_2012.loc[(hcris_2012[f'quartile_{i}'] == 1) & (hcris_2012['penalty'] == 0), 'price'].mean()
    Avg_per_group.append({'Quartile': i, 'Penalized_Mean_Price': round(treated_mean, 2), 'Non_penalized_Mean_Price': round(control_mean, 2)})

results_df = pd.DataFrame(Avg_per_group)
print(results_df.to_string(index=False))

# Q 7
hcris_2012['bed_quartile'] = pd.qcut(hcris_2012['beds'], 4, labels=False)
bed_quart_dummies = pd.get_dummies(hcris_2012['bed_quartile'], prefix='bed_quart').iloc[:, :-1] * 1
bed_quart_dummies = bed_quart_dummies.sub(bed_quart_dummies.mean(axis=0), axis=1)

Y = hcris_2012['price'].values
D = hcris_2012['penalty'].values
X = bed_quart_dummies.values

results = pd.DataFrame(index=['ATE', 'SE'], columns=['INV', 'MAH', 'IPW', 'OLS'])
cm = CausalModel(Y=Y, D=D, X=X)

cm.est_via_matching(weights='inv', matches=1, bias_adj=True)
inv_ate = cm.estimates['matching']['ate']
inv_se = cm.estimates['matching']['ate_se']
results.loc['ATE', 'INV'] = inv_ate
results.loc['SE', 'INV'] = inv_se


cm.est_propensity()
cm.est_via_weighting()
results.loc['ATE', 'IPW'] = cm.estimates['weighting']['ate']
results.loc['SE', 'IPW'] = cm.estimates['weighting']['ate_se']

cm.est_via_ols(adj=2)
results.loc['ATE', 'OLS'] = cm.estimates['ols']['ate']
results.loc['SE', 'OLS'] = cm.estimates['ols']['ate_se']

results = results.astype(float).round(2)
print(results)