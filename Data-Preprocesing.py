import pandas as pd
from datasets import load_dataset
import numpy as np

# Load the dataset from the hugging face dataset
dataset = load_dataset('c3po-ai/edgar-corpus')

# Convert the splits to DataFrames
df_train = pd.DataFrame(dataset['train'])
df_validation = pd.DataFrame(dataset['validation'])
df_test = pd.DataFrame(dataset['test'])

# Concatenate all splits
all_data = pd.concat([df_train, df_validation, df_test])

# Convert 'cik' to string
all_data['cik'] = all_data['cik'].astype(str)

# Load the fraud companies
fraud_companies = pd.read_csv("C:/DissData/Dissertation-Brunel/fraud_companies_with_cik.csv")
fraud_companies['CIK Number'] = fraud_companies['CIK Number'].astype(str)

df1 = all_data

df2 = fraud_companies

matching_cik = df1['cik'].isin(df2['CIK Number'])

# Matching ciks with the our dataset with the hugging face dataset

df1_with_matching_cik = df1[matching_cik]
df1_with_non_matching_cik = df1[~matching_cik]

df1_with_matching_cik['cik'].nunique()     # We got 85 unique ciks
df1_with_non_matching_cik['cik'].nunique() # We got 37924 unique ciks

# Filter df2 to only those rows where 'CIK Number' is in 'cik' of df1
df2_matching_with_df1 = df2[df2['CIK Number'].isin(df1['cik'])]

df2 = df2_matching_with_df1 # df2 is now the same as df1_with_matching_cik

# Checking If there are duplicat cik in our final dataset
duplicates_df2 = df2.duplicated(subset=['CIK Number'], keep = False)
num_duplicates_df2 = duplicates_df2.sum()

duplicates_df2 = df2.duplicated(subset=['CIK Number'], keep=False)
duplicate_rows_df2 = df2[duplicates_df2]
print(duplicate_rows_df2) # we got some duplicate ciks

df2.drop([124,81,206,215,98,195], inplace=True) # dropping the (124,81,206,215, 98 and 195)th rows as they were redundunt 

#df2.to_csv("Final_company_with_cik.csv", index= False)

merged_df = df1.merge(df2, how='inner', left_on='cik', right_on='CIK Number')

merged_df['Fraud'] = 'Yes'

# Find all unique cik in df1 that are not in df2
cik_df1 = set(df1['cik'].unique())
cik_df2 = set(df2['CIK Number'].unique())
cik_df1_not_in_df2 = list(cik_df1 - cik_df2)


# Randomly select 85 unique companies
random_cik = np.random.choice(cik_df1_not_in_df2, 85, replace=False)

# Select all rows from df1 that belong to these companies
new_rows = df1[df1['cik'].isin(random_cik)]

# Add 'Fraud' column and set it to 'No'
new_rows = new_rows.assign(Fraud='No')

# Append these rows to the final DataFrame
final_df = merged_df.append(new_rows, ignore_index=True)

final_df['Fraud'].value_counts()  # we  got Yes: 1660,  No: 436 in the Fraud column

#final_df.to_csv("Final_Dataset.csv")



