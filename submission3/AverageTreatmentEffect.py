import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


HCRIS_data = pd.read_csv('submission3/data/output/HCRIS_Data.csv')
HCRIS_data = HCRIS_data[HCRIS_data['year'] == 2012]

print(HCRIS_data.columns)

HCRIS_data.groupby(['provider_number'])