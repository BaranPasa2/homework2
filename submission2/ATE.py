import numpy as np
import pandas as pd
import seaborn as sns

HCRIS_data = pd.read_csv('/Users/baranpasa/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Emory/Junior Year/Junior Spring/ECON 470/ECON 470 Python /homework2/submission1/data/output/HCRIS_Data.csv')
HCRIS_data = HCRIS_data[HCRIS_data['year'] == 2012]

print(HCRIS_data.columns)

print(HCRIS_data['hrrp_payment'].unique())
print(HCRIS_data['hvbp_payment'].unique())

