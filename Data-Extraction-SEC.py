import pandas as pd
from fuzzywuzzy import fuzz, process

from datasets import load_dataset

dataset = load_dataset('c3po-ai/edgar-corpus')


# Convert the training split to a DataFrame
df_train = pd.DataFrame(dataset['train'])

# Convert the validation split to a DataFrame
df_validation = pd.DataFrame(dataset['validation'])

# Convert the test split to a DataFrame
df_test = pd.DataFrame(dataset['test'])

'''
file_path = "C://DissData/sec.gov_Archives_edgar_cik-lookup-data.txt"

# Read the file line by line
with open(file_path, 'r') as file:
    lines = file.readlines()
    
fraud_companies = pd.read_excel("C://DissData//Fraud_Companies_dataset.xlsx")

data_list = [line.rsplit(":", 2)[:2] for line in lines]

# Create a DataFrame
df = pd.DataFrame(data_list, columns=["Company Name", "CIK Number"])


cik_dict = pd.Series(df["CIK Number"].values, index=df["Company Name"]).to_dict()

# Function to match company names and return CIK number
def match_name(name, list_names, min_score=0):
    # -1 score incase we don't get any match
    max_score = -1
    # Returning empty name for no match as well
    max_name = ""
    # Iternating over all names in the other
    for name2 in list_names:
        #Finding fuzzy match score
        score = process.extractOne(name, [name2], score_cutoff=min_score)
        # Checking if we are above our threshold and have a better score
        if score and score[1] > max_score:
            max_name = name2
            max_score = score[1]
    return cik_dict.get(max_name)

# Use the function to match the company names and get the CIK number
fraud_companies["CIK Number"] = fraud_companies["Company Name"].apply(lambda x: match_name(x, cik_dict.keys(), 70))


fraud_companies_to_save = fraud_companies[["Company Name", "CIK Number"]]

# Save to CSV
fraud_companies_to_save.to_csv("fraud_companies_with_cik.csv", index=False)

'''
all_data = pd.concat([df_train, df_validation, df_test])

all_data['cik'] = all_data['cik'].astype(str)

fraud_companies = pd.read_csv("fraud_companies_with_cik.csv")

'''
fraud_companies['CIK Number'] = fraud_companies['CIK Number'].astype(str)

# Merge the filings data with your existing DataFrame
merged_df = pd.merge(fraud_companies, all_data, left_on="CIK Number", right_on="cik", how="left")

merged_df.to_csv("fraud_companies_with_cik_and_filings.csv", index=False)
'''
fraud_companies['CIK Number'] = fraud_companies['CIK Number'].astype(str)

# Now perform the merge
merged_df = pd.merge(fraud_companies, all_data, left_on="CIK Number", right_on="cik", how="left")

# Drop the 'cik' column as it's redundant
merged_df = merged_df.drop(columns=['cik'])

merged_df['Fraud'] = 'Yes'

# Save to CSV
#merged_df.to_csv("fraud_companies_with_cik_and_filings.csv", index=False)

# Number of unique non-fraud companies to select
num_non_fraud = fraud_companies['CIK Number'].nunique()

# Get a DataFrame of non-fraud companies
non_fraud = all_data[~all_data['cik'].isin(fraud_companies['CIK Number'])]

# Randomly select num_non_fraud unique companies
non_fraud_sample_ciks = non_fraud['cik'].drop_duplicates().sample(n=num_non_fraud, random_state=1)

# Get all the years of data for the selected non-fraud companies
non_fraud_sample = non_fraud[non_fraud['cik'].isin(non_fraud_sample_ciks)]

# Add a 'Fraud' column and set it to 'No'
non_fraud_sample['Fraud'] = 'No'

# Concatenate the fraud and non-fraud DataFrames
all_companies = pd.concat([merged_df, non_fraud_sample])

# Save to CSV
all_companies.to_csv("final_dataset.csv", index=False)

