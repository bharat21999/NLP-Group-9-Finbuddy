import pandas as pd

# Load both datasets
capital_df = pd.read_csv("/home/ubuntu/finbuddy/Data/capital_com_financial_dictionary.csv")
investopedia_df = pd.read_csv("/home/ubuntu/finbuddy/Data/investopedia_terms_final.csv")

# Basic cleanup (strip spaces)
capital_df['term'] = capital_df['term'].str.strip()
investopedia_df['term'] = investopedia_df['term'].str.strip()

# Set 'term' as index for both
capital_df.set_index('term', inplace=True)
investopedia_df.set_index('term', inplace=True)

# Identify overlapping terms
overlapping_terms = set(capital_df.index).intersection(set(investopedia_df.index))

# Report how many overlaps (capital will replace investopedia in these cases)
print(f" Found {len(overlapping_terms)} overlapping terms where Capital.com definitions will be preferred.")

# Merge with priority to Capital.com
merged_df = capital_df.combine_first(investopedia_df)

# Reset index to normal
merged_df = merged_df.reset_index()

# Save merged dataset
merged_df.to_csv("/home/ubuntu/finbuddy/Data/merged_financial_terms.csv", index=False)

print(f" Successfully merged {len(merged_df)} terms and saved to merged_financial_terms.csv")
