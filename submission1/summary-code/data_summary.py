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
price_col = "estimated_price"  # Replace with actual column name

# Remove negative and extreme outliers (using IQR method)
Q1 = HCRIS_data[price_col].quantile(0.25)
Q3 = HCRIS_data[price_col].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

cleaned_HCRIS = HCRIS_data[(HCRIS_data[price_col] >= lower_bound) & (HCRIS_data[price_col] <= upper_bound)]

# Plot violin plot
plt.figure(figsize=(12, 6))
sns.violinplot(x=cleaned_HCRIS['year'], y=cleaned_HCRIS[price_col], inner="quartile", palette="pastel")
plt.xlabel("Year")
plt.ylabel("Estimated Prices")
plt.title("Distribution of Estimated Prices Per Year (Outliers Removed)")
plt.xticks(rotation=45)
plt.grid()
plt.show()
