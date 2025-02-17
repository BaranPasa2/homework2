import pandas as pd
import numpy
import matplotlib.pyplot as plt
import seaborn as sns


HCRIS_data = pd.read_csv('submission1/data/output/HCRIS_Data.csv')

# print(HCRIS_data['provider_number'].count())


# Point 1
hospital_reports = HCRIS_data.groupby(['year', 'provider_number']).size().reset_index(name='report_count')
hospital_reports_multi = hospital_reports[hospital_reports['report_count'] > 0]
hospital_reports_multi = hospital_reports_multi.groupby('year')['provider_number'].nunique().reset_index()

# Point 2
uniqueCodes = hospital_reports_multi['provider_number'].nunique()
print(f"Total number of unique hospitals: {uniqueCodes}")


# Point 1 Graphs
plt.figure(figsize=(10, 5))
sns.lineplot(x='year', y='provider_number', data=hospital_reports_multi, marker="o")
plt.xlabel("Year")
plt.ylabel("Number of Hospitals with Multiple Reports")
plt.title("Hospitals Filing More Than One Report Per Year")
plt.grid()
plt.show()

# Point 3 - Getting a key error issue. 

plt.figure(figsize=(12, 6))
sns.violinplot(x=HCRIS_data['year'], y=HCRIS_data['tot_charges'], inner='quartile', palette='muted')
plt.xlabel("Year")
plt.ylabel("Total Charges")
plt.title("Distribution of Total Charges per Year")
plt.xticks(rotation=45)
plt.grid()
plt.show()

# Part 4
# Ensure correct column name for estimated prices


# Ensure correct column names (update if needed)
df = HCRIS_data
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
df["discount_factor"] = 1 - df[tot_discounts_col] / df[tot_charges_col]
df["price_num"] = (df[ip_charges_col] + df[icu_charges_col] + df[ancillary_charges_col]) * df["discount_factor"] - df[tot_mcare_payment_col]
df["price_denom"] = df[tot_discharges_col] - df[mcare_discharges_col]

# Avoid division by zero
df["estimated_price"] = df["price_num"] / df["price_denom"]
df = df.replace([float('inf'), -float('inf')], None).dropna(subset=["estimated_price"])

# Step 2: Remove negative and extreme outliers using IQR
Q1 = df["estimated_price"].quantile(0.25)
Q3 = df["estimated_price"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

filtered_df = df[(df["estimated_price"] >= lower_bound) & (df["estimated_price"] <= upper_bound)]

# Step 3: Plot violin plot
plt.figure(figsize=(12, 6))
sns.violinplot(x=filtered_df[year_col], y=filtered_df["estimated_price"], inner="quartile", palette="pastel")
plt.xlabel("Year")
plt.ylabel("Estimated Price")
plt.title("Distribution of Estimated Prices Per Year (Outliers Removed)")
plt.xticks(rotation=45)
plt.grid()
plt.show()
