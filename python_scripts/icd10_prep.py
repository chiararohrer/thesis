import numpy as np
import pandas as pd
import random
import yaml
from preprocessing_functions import find_disease, find_sr_disease


phenotypes_path = '/path_to_data/ukb677323.tab.gz'
olink_path = '/path_to_data/olink_data.txt.gz'
chunks = 10000
random.seed(10)

# Create a dummy df to get all column names of the raw phenotype data
dummy = pd.DataFrame()
for chunk in pd.read_csv(phenotypes_path, chunksize=10, sep='\t', compression='gzip'):
    dummy = pd.concat([dummy, chunk])
    break

# Filter out al column names that start with 41280 (Date of first in-patient diagnosis - ICD10) and save them to tsv file
date_columns = dummy.filter(like='f.41280.')
date_list = date_columns.columns.tolist()
date_df = pd.DataFrame({'cols': date_list})
date_df.to_csv(f'/mypath/diseases/icd10_date_cols.tsv', sep='\t', index=False, header=False)

# Filter out all column names that start with 41270 (Diagnoses - ICD10) and save them to tsv file
code_columns = dummy.filter(like='f.41270.')
code_list = code_columns.columns.tolist()
code_df = pd.DataFrame({'cols': code_list})
code_df.to_csv(f'/mypath/diseases/icd10_code_cols.tsv', sep='\t', index=False, header=False)

# Filter out all column names that start with 20002.0. (self-reported conditions (except cancers) at BL)
sr_columns = dummy.filter(like='f.20002.0.')
sr_list = sr_columns.columns.tolist()
sr_df = pd.DataFrame({'cols': sr_list})

# Filter out all column names that start with 20001.0. (self-reported cancers at BL)
sr_cancer_columns = dummy.filter(like='f.20001.0.')
sr_cancer_list = sr_cancer_columns.columns.tolist()
sr_cancer_df = pd.DataFrame({'cols': sr_cancer_list})

# Given the wanted column names, load complete phenotype data but only wanted (code + date) columns:
cols = ['f.eid'] + code_list + date_list + sr_list + sr_cancer_list
icd_df = pd.DataFrame()
for chunk in pd.read_csv(phenotypes_path, chunksize=chunks, sep='\t', compression='gzip', usecols=cols):
    icd_df = pd.concat([icd_df, chunk])

# Get ICD codes for all diseases and for every disease, filter out diagnosed patients, their ICD10 code and first occurrence date. If multiple for one patient, take only first diagnosis.
with open('/mypath/diseases/icd10_diseases_v2.yaml', 'r') as file:
    icd10_diseases = yaml.safe_load(file)

with open('/mypath/diseases/self_reported_codes_v2.yaml', 'r') as file:
    sr_diseases = yaml.safe_load(file)

disease_names = list(icd10_diseases.keys())
for disease in disease_names:
    #if not os.path.isfile(f'/mypath/diseases/icd10_filtered/filtered_{disease}.tsv'):
    if True:
        specific_list = list(icd10_diseases[disease])
        print(disease, specific_list)
        returned_df = find_disease(icd_df[['f.eid']+ code_list + date_list], specific_list)
        returned_df.to_csv(f'/mypath/diseases/icd10_filtered/filtered_{disease}.tsv', sep='\t', index=False, header=True)
    #if not os.path.isfile(f'/mypath/diseases/icd10_filtered/self_reported_{disease}.tsv'):
    if True:
        if disease != 'non-hodgkin lymphoma' and disease != 'multiple myeloma and malignant plasma cell neoplasms':
            self_reported_df = find_sr_disease(icd_df[['f.eid']+sr_list], list(sr_diseases[disease]))
        else:
            self_reported_df = find_sr_disease(icd_df[['f.eid']+sr_cancer_list], list(sr_diseases[disease]))
        self_reported_df.to_csv(f'/mypath/diseases/icd10_filtered/self_reported_{disease}.tsv', sep='\t', index=False, header=True)