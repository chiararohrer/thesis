import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LinearRegression
import sys
import os
import torch
import yaml
from preprocessing_functions import load_pt, write_pt
from move.data.preprocessing import standardize


def residualize(path, config_name, size, run=None):
    # open old config file and extract categorical and continuous data file names 
    try:
        with open(f'{path}/config/data/{config_name}.yaml', 'r') as file:
            old_yaml = yaml.safe_load(file)
    except:
        with open(f'{path}/config/data/{config_name}0.yaml', 'r') as file:
            old_yaml = yaml.safe_load(file)
    cat = [item['name'] for item in old_yaml['categorical_inputs']]
    cont = [item['name'] for item in old_yaml['continuous_inputs']]
    
    # load all data files that are considered covariates, i.e. have to be rediualized out
    interim_path = f'{path}/interim_data_nores_{size}_{run}' 

    encoded_age_df, age_fn, age_map, age_shape = load_pt(interim_path, f'age_{size}')
    encoded_fasting_df, fasting_fn, fasting_map, fasting_shape = load_pt(interim_path, f'fasting_time_{size}')
    encoded_sex_df, sex_fn, sex_map, sex_shape = load_pt(interim_path, f'sex_{size}')
    encoded_ethnicity_df, ethnicity_fn, ethnicity_map, ethnicity_shape = load_pt(interim_path, f'ethnic_background_{size}')
    encoded_centre_df, centre_fn, centre_map, centre_shape = load_pt(interim_path, f'assessment_centre_{size}')
    encoded_sm_df, sm_fn, sm_map, sm_shape = load_pt(interim_path, f'smoking_status_{size}')
    encoded_alc_df, alc_fn, alc_map, alc_shape = load_pt(interim_path, f'alcohol_frequency_{size}')

    # Concatenate all the covariate features
    covariates = pd.concat([encoded_age_df, encoded_fasting_df, encoded_sex_df, encoded_ethnicity_df, encoded_centre_df, encoded_sm_df, encoded_alc_df], axis=1)

    # load all the data files that are transformed to residuals
    encoded_urine_df, urine_fn, urine_map, urine_shape = load_pt(interim_path, f'urine_{size}')
    encoded_bb_df, bb_fn, bb_map, bb_shape = load_pt(interim_path, f'blood_biochemistry_{size}')
    encoded_bc_df, bc_fn, bc_map, bc_shape = load_pt(interim_path, f'blood_count_{size}')
    ### DO NOT RESIDUALIZE GENOMICS
    #encoded_gen_df, gen_fn, gen_map, gen_shape = load_pt(interim_path, f'genomics_{size}')
    encoded_inf_df, inf_fn, inf_map, inf_shape = load_pt(interim_path, f'infectious_diseases_{size}')
    encoded_metabolomics_df, met_fn, met_map, met_shape = load_pt(interim_path, f'metabolomics_{size}')
    encoded_physical_df, physical_fn, physical_map, physical_shape = load_pt(interim_path, f'physical_measurements_{size}')
    encoded_proteomics_df, proteomics_fn, proteomics_map, proteomics_shape = load_pt(interim_path, f'proteomics_{size}')


    # Concatenate them into one df [GENOMICS REMOVED]
    features_cont = pd.concat([encoded_urine_df, encoded_bb_df, encoded_bc_df, encoded_metabolomics_df, encoded_physical_df, encoded_proteomics_df, encoded_inf_df], axis=1)

    idx = torch.load(f'{interim_path}/indices.pt')
    train_id = idx['train_indices']
    test_id = idx['test_indices']

    raw_data_path = old_yaml['raw_data_path']
    old_ids = pd.read_csv(f'{raw_data_path}/id_{size}.txt', sep=' ', header=None)
    old_ids.columns = ['f.eid']
    
    features_cont, covariates = pd.concat([old_ids, features_cont], axis=1), pd.concat([old_ids, covariates], axis=1)
    features_cont.replace(0, np.nan, inplace=True)

    features = features_cont
    
    train_features = features.iloc[train_id.tolist(), :]
    test_features = features.iloc[test_id.tolist(), :]
    train_covariates = covariates.iloc[train_id.tolist(), :]
    test_covariates = covariates.iloc[test_id.tolist(), :]

    # Compute all the residuals
    residuals = pd.DataFrame(index=features.index, columns=features.columns)
    residuals['f.eid'] = features['f.eid']

    for feature in features.columns[1:]:  
        not_nan_mask = features[feature].notna()
        train_not_nan_mask = train_features[feature].notna()
        if train_not_nan_mask.any():  
            model = LinearRegression()
            model.fit(train_covariates[train_not_nan_mask], train_features.loc[train_not_nan_mask, feature])
            predictions = model.predict(covariates[not_nan_mask])
            res = features.loc[not_nan_mask, feature] - predictions
            residuals.loc[not_nan_mask, feature] = res
            residuals.loc[~not_nan_mask, feature] = np.NaN
        else:
            residuals[feature] = features[feature]
    
    
    
    residuals_np = residuals.iloc[:,1:].to_numpy(na_value=np.NaN)
    
    # CATCH EXTREME VALUES BEFORE RES BY CLIPPING:
    #residuals_np = np.clip(residuals_np, -100, 100)

    residuals_np = standardize(residuals_np, train_id)
    
    residuals.iloc[:,1:] = residuals_np
    
    # CATCH EXTREME VALUES AFTER RES:
    #residuals_np = np.where(residuals_np > 100, 100, residuals_np)
    #residuals_np = np.where(residuals_np < -100, -100, residuals_np)
    #residuals.iloc[:, 1:] = residuals.iloc[:, 1:].where((residuals.iloc[:, 1:] < 150) & (residuals.iloc[:, 1:] > -150), 0)

    # Write the residuals to a new folder, and format them the same as the input data
    interim_res_path = f'{path}/interim_data_res_{size}_{run}'
    
    print('residuals to be written to files: ', residuals)
    os.makedirs(f'{interim_res_path}/', exist_ok=True)
    write_pt(interim_res_path, f'urine_{size}', encoded_urine_df, residuals, urine_fn, urine_map, False, urine_shape)
    write_pt(interim_res_path, f'blood_biochemistry_{size}', encoded_bb_df, residuals, bb_fn, bb_map, False, bb_shape)
    write_pt(interim_res_path, f'blood_count_{size}', encoded_bc_df, residuals, bc_fn, bc_map, False, bc_shape)
    
    ### GENOMICS: LOAD AND SAVE FILE DIRECTLY
    #write_pt(interim_res_path, f'genomics_{size}', encoded_gen_df, residuals, gen_fn, gen_map, False, gen_shape)
    copy_gen = torch.load(f'{interim_path}/genomics_{size}.pt')
    torch.save(obj=copy_gen, f=f'{interim_res_path}/genomics_{size}.pt')
    copy_pqtl = torch.load(f'{interim_path}/pQTL_{size}.pt')
    torch.save(obj=copy_pqtl, f=f'{interim_res_path}/pQTL_{size}.pt')

    write_pt(interim_res_path, f'infectious_diseases_{size}', encoded_inf_df, residuals, inf_fn, inf_map, False, inf_shape)
    write_pt(interim_res_path, f'metabolomics_{size}', encoded_metabolomics_df, residuals, met_fn, met_map, False, met_shape)
    write_pt(interim_res_path, f'physical_measurements_{size}', encoded_physical_df, residuals, physical_fn, physical_map, False, physical_shape)
    write_pt(interim_res_path, f'proteomics_{size}', encoded_proteomics_df, residuals, proteomics_fn, proteomics_map, False, proteomics_shape)
    
    # transfer indices file to interim_data_res folder
    idx = torch.load(f'{interim_path}/indices.pt')  
    torch.save(obj=idx, f=f'{interim_res_path}/indices.pt')


if __name__ == "__main__":
    path = sys.argv[1]
    config_name = sys.argv[2]
    size = int(sys.argv[3])
    run = sys.argv[4]
    residualize(path, config_name, size, run)