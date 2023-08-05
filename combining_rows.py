# Importing the required libraries
import pandas as pd

# Load the dataset
df = pd.read_csv('Final_Dataset.csv')

df['year'] = df['year'].astype(str)
# Define a custom function for joining
def join_strings(series):
    return ''.join([str(item) for item in series])

# Perform the same grouping and aggregation as before, using the new join_strings function
merged_df = df.groupby(['Company Name', 'CIK Number']).agg({
    'year': lambda x: ', '.join(x.unique()), # join unique years
    'section_1': join_strings, # join section texts
    'section_1A': join_strings,
    'section_1B': join_strings if 'section_1B' in df.columns else None,
    'section_2': join_strings,
    'section_3': join_strings,
    'section_4': join_strings,
    'section_5': join_strings,
    'section_6': join_strings,
    'section_7': join_strings,
    'section_7A': join_strings if 'section_7A' in df.columns else None,
    'section_8': join_strings,
    'section_9': join_strings,
    'section_9A': join_strings if 'section_9A' in df.columns else None,
    'section_9B': join_strings,
    'section_10': join_strings,
    'section_11': join_strings,
    'section_12': join_strings,
    'section_13': join_strings,
    'section_14': join_strings,
    'section_15': join_strings,
    'Fraud': lambda x: 'Yes' if x.value_counts().idxmax() == 'Yes' else 'No' # take the most frequent value
})

# Filter out None values from the agg dictionary (for non-existing columns)
merged_df = merged_df.loc[:, merged_df.columns.notnull()]

# Reset the index to make 'Company Name' and 'CIK Number' normal columns
merged_df.reset_index(inplace=True)

print(merged_df.head(5))

#merged_df.to_csv("Fraud_Comapnies_only.csv" , index=False)