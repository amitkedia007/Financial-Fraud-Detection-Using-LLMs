# This file contains the loading of the dataset from the edgar corpus in hugging face dataset.
# Matching the existing data of company names with the name of the company from the edgar corpus using the fuzzy logic.
# Appending the CIK number to the company name.


import argparse
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


parser = argparse.ArgumentParser(description="Match company names to CIK numbers")
parser.add_argument("--cik_lookup", default="sec.gov_Archives_edgar_cik-lookup-data.txt",
                    help="Path to CIK lookup text file")
parser.add_argument("--fraud_excel", default="Fraud_Companies_dataset.xlsx",
                    help="Path to Excel file containing fraud companies")
parser.add_argument("--output", default="fraud_companies_with_cik.csv",
                    help="Path for output CSV")
args = parser.parse_args()

file_path = args.cik_lookup

# Read the file line by line
with open(file_path, 'r') as file:
    lines = file.readlines()

fraud_companies = pd.read_excel(args.fraud_excel)

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
fraud_companies_to_save.to_csv(args.output, index=False)



