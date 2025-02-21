import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import mahalanobis
import statsmodels.api as sm

HCRIS_data = pd.read_csv('submission3/data/output/HCRIS_Data.csv')

HCRIS_data['year'] = pd.to_numeric(HCRIS_data['year'], errors='coerce')
HCRIS_data['hvbp_payment'] = pd.to_numeric(HCRIS_data['hvbp_payment'], errors='coerce')
HCRIS_data['hrrp_payment'] = pd.to_numeric(HCRIS_data['hrrp_payment'], errors='coerce')
HCRIS_data['tot_charges'] = pd.to_numeric(HCRIS_data['tot_charges'], errors='coerce')
HCRIS_data['tot_discharges'] = pd.to_numeric(HCRIS_data['tot_discharges'], errors='coerce')
HCRIS_data['beds'] = pd.to_numeric(HCRIS_data['beds'], errors='coerce')

HCRIS_2011 = HCRIS_data[HCRIS_data['year'] == 2011]

HCRIS_2011 = HCRIS_2011.dropna(subset=['hvbp_payment', 'hrrp_payment', 'tot_charges', 'tot_discharges'])

HCRIS_2011['penalty'] = (HCRIS_2011['hvbp_payment'] + HCRIS_2011['hrrp_payment']) < 0

HCRIS_2011 = HCRIS_2011[HCRIS_2011['tot_discharges'] > 0].copy()

HCRIS_2011['average_price'] = HCRIS_2011['tot_charges'] / HCRIS_2011['tot_discharges']

avgPenPrice = HCRIS_2011.groupby('penalty')['average_price'].mean()


# Filter dataset for the year 2010 and drop missing 'beds' values
HCRIS_Beds = HCRIS_2011.dropna(subset=['beds'])

# Ensure 'beds' is numeric
HCRIS_Beds['beds'] = pd.to_numeric(HCRIS_Beds['beds'], errors='coerce')

# Create quartiles for bed size
HCRIS_Beds['bed_quartile'] = pd.qcut(HCRIS_Beds['beds'], q=4, labels=["Q1", "Q2", "Q3", "Q4"])

# Create 4 indicator variables for each quartile
HCRIS_Beds['quartile_1'] = (HCRIS_Beds['bed_quartile'] == "Q1").astype(int)
HCRIS_Beds['quartile_2'] = (HCRIS_Beds['bed_quartile'] == "Q2").astype(int)
HCRIS_Beds['quartile_3'] = (HCRIS_Beds['bed_quartile'] == "Q3").astype(int)
HCRIS_Beds['quartile_4'] = (HCRIS_Beds['bed_quartile'] == "Q4").astype(int)

# Define penalty as whether the sum of HRRP and HVBP payments is negative
#HCRIS_Beds['penalty'] = (HCRIS_Beds['hvbp_payment'] + HCRIS_Beds['hrrp_payment']) < 0

# Remove rows where total discharges is zero to avoid division errors

# Calculate average price safely
#HCRIS_Beds['average_price'] = HCRIS_Beds['tot_charges'] / HCRIS_Beds['tot_discharges']

# Group by penalty status and quartile, then calculate mean average price
quartile_price_table = HCRIS_Beds.groupby(['bed_quartile', 'penalty'])['average_price'].mean().unstack()

# Create a heatmap visualization using seaborn
plt.figure(figsize=(10, 6))
sns.heatmap(quartile_price_table, annot=True, fmt=".2f", cmap="viridis", linewidths=0.5)

# Customize the plot
plt.title("Average Price by Bed Quartile and Penalty Status (2010)")
plt.xlabel("Penalty Status")
plt.ylabel("Bed Quartile")
plt.xticks(ticks=[0, 1], labels=["Not Penalized", "Penalized"], rotation=0)
plt.yticks(rotation=0)

# Show the plot
plt.show()

print(HCRIS_Beds.head())

# 1. Nearest Neighbor Matching (1-to-1) with Inverse Variance Distance
# Compute variance of bed sizes within each quartile
quartile_variance = HCRIS_Beds.groupby('bed_quartile')['beds'].var()

# Split into treated (penalized) and control groups
treated = HCRIS_Beds[HCRIS_Beds['penalty'] == 1].copy()
control = HCRIS_Beds[HCRIS_Beds['penalty'] == 0].copy()

# Find nearest neighbor using inverse variance distance (Euclidean approximation)
nbrs = NearestNeighbors(n_neighbors=1, metric='euclidean')
nbrs.fit(control[['beds']])
distances, indices = nbrs.kneighbors(treated[['beds']])

# Compute ATT (average treatment effect on the treated)
matched_controls = control.iloc[indices.flatten()]
ate_inverse_variance = (treated['average_price'].values - matched_controls['average_price'].values).mean()
