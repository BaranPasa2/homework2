import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


HCRIS_data = pd.read_csv('/Users/baranpasa/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Emory/Junior Year/Junior Spring/ECON 470/ECON 470 Python /homework2/submission3/data/output/HCRIS_Data.csv')
# print(HCRIS_data['provider_number'].count())



HCRISS1996 = pd.read_csv("/Users/baranpasa/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Emory/Junior Year/Junior Spring/ECON 470/ECON 470 Python /homework2/submission3/data/output/HCRIS_v1996.csv")
HCRISS2010 = pd.read_csv("/Users/baranpasa/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Emory/Junior Year/Junior Spring/ECON 470/ECON 470 Python /homework2/submission3/data/output/HCRIS_v2010.csv")
HCRIS_total = pd.concat([HCRISS1996, HCRISS2010], ignore_index=False)

hospital_charge_counts = HCRIS_total.groupby(['year', 'provider_number']).size().reset_index(name='charge_count')
multiple_charges = hospital_charge_counts[hospital_charge_counts['charge_count'] > 1]
hospitals_per_year = multiple_charges.groupby('year')['provider_number'].nunique().reset_index()

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(hospitals_per_year['year'], hospitals_per_year['provider_number'], marker='o', linestyle='-')
plt.xlabel('Year')
plt.ylabel('Number of Hospitals with Multiple Charges')
plt.title('Hospitals with Multiple Charges Per Year')
plt.grid(True)
plt.show()

# Point 2
hospital_reports = HCRIS_data.groupby(['year', 'provider_number']).size().reset_index(name='report_count')
hospital_reports_multi = hospital_reports[hospital_reports['report_count'] > 0]
hospital_reports_multi = hospital_reports_multi.groupby('year')['provider_number'].nunique().reset_index()

unique_hospitals_per_year = hospital_reports_multi.groupby('year')['provider_number'].unique()
plt.figure(figsize=(10, 5))
plt.plot(unique_hospitals_per_year.index, unique_hospitals_per_year.values, marker='o', linestyle='-')
plt.xlabel("Year")
plt.ylabel("Number of Unique Hospitals")
plt.title("Number of Unique Hospitals Per Year")
plt.grid(True)
plt.show()

# Point 3 - Getting a key error issue. 
# Ensure no non-positive values before applying log
HCRIS_data = HCRIS_data[HCRIS_data['tot_charges'] > 0]  # Remove non-positive values to prevent -inf or NaN
HCRIS_data['log_charges'] = np.log(HCRIS_data['tot_charges'])
HCRIS_data = HCRIS_data.replace([np.inf, -np.inf], np.nan).dropna(subset=['log_charges'])

plt.figure(figsize=(12, 6))
sns.violinplot(x=HCRIS_data['year'], y=HCRIS_data['log_charges'], inner='quartile', palette='muted')
plt.xlabel("Year")
plt.ylabel("Total Charges")
plt.title("Distribution of Total Charges per Year")
plt.xticks(rotation=45)
plt.grid()
plt.show()

# Part 4
# Ensure correct column name for estimated prices


# Ensure correct column names (update if needed)

tot_discounts_col = "tot_discounts"
tot_charges_col = "tot_charges"
ip_charges_col = "ip_charges"
icu_charges_col = "icu_charges"
ancillary_charges_col = "ancillary_charges"
tot_mcare_payment_col = "tot_mcare_payment"
tot_discharges_col = "tot_discharges"
mcare_discharges_col = "mcare_discharges"
year_col = "year"


# Step 1: Compute estimated price using the updated formula
HCRIS_data["discount_factor"] = 1 - HCRIS_data[tot_discounts_col] / HCRIS_data[tot_charges_col]
HCRIS_data["price_num"] = (HCRIS_data[ip_charges_col] + HCRIS_data[icu_charges_col] + HCRIS_data[ancillary_charges_col]) * HCRIS_data["discount_factor"] - HCRIS_data[tot_mcare_payment_col]
HCRIS_data["price_denom"] = HCRIS_data[tot_discharges_col] - HCRIS_data[mcare_discharges_col]

# Avoid division by zero
HCRIS_data["estimated_price"] = HCRIS_data["price_num"] / HCRIS_data["price_denom"]
HCRIS_data = HCRIS_data.replace([float('inf'), -float('inf')], None).dropna(subset=["estimated_price"])

HCRIS_data["estimated_price"] = pd.to_numeric(HCRIS_data["estimated_price"], errors="coerce")

# Ensure year column is string (categorical)
HCRIS_data[year_col] = HCRIS_data[year_col].astype(str)

# Remove NaN or Inf values
HCRIS_data = HCRIS_data.replace([np.inf, -np.inf], np.nan).dropna(subset=["estimated_price"])

# Filter outliers
Q1 = HCRIS_data["estimated_price"].quantile(0.25)
Q3 = HCRIS_data["estimated_price"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

HCRIS_data_filtered = HCRIS_data[(HCRIS_data["estimated_price"] >= lower_bound) & (HCRIS_data["estimated_price"] <= upper_bound)]

# Plot violin plot
plt.figure(figsize=(12, 6))
sns.violinplot(x=HCRIS_data_filtered[year_col], y=HCRIS_data_filtered["estimated_price"], inner="quartile", palette="pastel")
plt.xlabel("Year")
plt.ylabel("Estimated Price")
plt.title("Distribution of Estimated Prices Per Year (Outliers Removed)")
plt.xticks(rotation=45)
plt.grid()
plt.show()