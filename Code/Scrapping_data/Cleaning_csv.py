
import pandas as pd

# Load your A–Z terms
terms_az = pd.read_csv("/home/ubuntu/finbuddy/Data/investopedia_terms.csv")

# Load the new number-based terms
terms_num = pd.read_csv("/home/ubuntu/finbuddy/Data/investopedia_num_terms.csv")

# Merge both datasets
merged_terms = pd.concat([terms_az, terms_num], ignore_index=True)

# Optional: Drop exact duplicates based on 'term' and 'definition'
merged_terms = merged_terms.drop_duplicates(subset=["term", "definition"])

# Save merged dataset
merged_terms.to_csv("/home/ubuntu/finbuddy/Data/investopedia_terms_merged.csv", index=False)

print(f" Merged dataset saved with {len(merged_terms)} terms at /home/ubuntu/finbuddy/Data/investopedia_terms_merged.csv")


import pandas as pd

# Load your cleaned file
df = pd.read_csv("/home/ubuntu/finbuddy/Data/investopedia_terms_merged.csv")

# Function to check if the **full term phrase** appears in the definition
def full_term_in_definition(row):
    term = str(row['term']).lower().strip()
    definition = str(row['definition']).lower().strip()
    return term in definition  # Check full phrase match (in order)

# Apply the stricter filter
df_final = df[df.apply(full_term_in_definition, axis=1)].copy()

# Optionally remove very short definitions
df_final = df_final[df_final['definition'].str.len() > 30]

# Save the final truly clean dataset
df_final.to_csv("/home/ubuntu/finbuddy/Data/investopedia_terms_final.csv", index=False)

print(f" Final clean terms (full phrase match): {len(df_final)} saved to investopedia_terms_final.csv")
