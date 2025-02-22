import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from causalinference import CausalModel 
import markdown as Markdown
import pandas as pd
import numpy as np

HCRIS_data = pd.read_csv('submission3/data/output/HCRIS_Data.csv')
#from data_summary_v3 import HCRIS_data_filtered as HCRIS_data

print(HCRIS_data.columns)

#HCRIS_data = HCRIS_data[HCRIS_data['year'] == 2012]
HCRIS_data['discount_factor'] = 1 - (HCRIS_data['tot_discounts'] / HCRIS_data['tot_charges'])
HCRIS_data['price_num'] = (HCRIS_data['ip_charges'] + HCRIS_data['icu_charges'] + HCRIS_data['ancillary_charges']) * HCRIS_data['discount_factor'] - HCRIS_data['tot_mcare_payment']
HCRIS_data['price_denom'] = HCRIS_data['tot_charges'] - HCRIS_data['mcare_discharges']
HCRIS_data['price'] = HCRIS_data['price_num']/HCRIS_data['price_denom']

final_hcris = HCRIS_data[
    (HCRIS_data['price_denom'] > 100) &
    (HCRIS_data['price_num'] >0) &
    (HCRIS_data['price'] <100000) &
    (HCRIS_data['beds'] > 30) &
    (HCRIS_data['year'] == 2012)
]

HCRIS_data = HCRIS_data[['hrrp_payment', 'hvbp_payment']].dropna()
print(HCRIS_data[HCRIS_data['hrrp_payment'] != 0])

# print(final_hcris[final_hcris ['penalty'] == 1])