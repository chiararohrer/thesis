import pandas as pd
import numpy as np
import os
import torch
import yaml
import move

def parental_history(df, parent):
    df['concatenated'] = df[df.columns[1:]].apply(lambda row: row.dropna().astype(int).tolist(), axis=1)
    df = df[['f.eid', 'concatenated']]
    matrix = pd.DataFrame(columns=['f.eid']+[f'{i}' for i in range(1,15)])
    matrix['f.eid'] = df['f.eid']
    matrix = pd.merge(matrix, df)
    for index, row in matrix.iterrows():
        for val in matrix.iat[index, -1]:
            if val > 0:
                matrix.iat[index, val] = 1
    matrix = matrix.fillna(0)
    matrix = matrix.drop(columns=['concatenated'])
    mapping = {
        '1': f'{parent}: Heart disease',
        '2': f'{parent}: Stroke',
        '3': f'{parent}: Lung cancer',
        '4': f'{parent}: Bowel cancer',
        '5': f'{parent}: Breast cancer',
        '6': f'{parent}: Chronic bronchitis/emphysema',
        '7': f'{parent}: ?',
        '8': f'{parent}: High blood pressure',
        '9': f'{parent}: Diabetes',
        '10': f'{parent}: Alzheimers disease/dementia',
        '11': f'{parent}: Parkinsons disease',
        '12': f'{parent}: Severe depression',
        '13': f'{parent}: Prostate cancer',
        '14': f'{parent}: Hip fracture',
    }
    matrix = matrix.rename(columns=mapping, inplace=False)
    matrix = matrix.loc[:, ~(matrix == 0).all()]
    return matrix


def get_id_from_tsv(path):
    ids = pd.read_csv(path, sep='\t')
    id_list = [str(ids['id'].values[i]) for i in range(len(ids['id']))]
    id_complete = ['f.eid']+['f.' + col + '.0.0' for col in id_list]
    return id_complete


def make_folders(foldername):
    if not os.path.exists(f'/mypath/{foldername}'):
        os.makedirs(f'/mypath/{foldername}/config/data/')
        os.makedirs(f'/mypath/{foldername}/config/experiment/')
        os.makedirs(f'/mypath/{foldername}/config/task/')
        os.makedirs(f'/mypath/{foldername}/data/')
        os.makedirs(f'/mypath/{foldername}/preprocessing/')


def sample_store_split(foldername, df_name, df, train_samples, test_samples = None):
    if 'f.eid' not in df.columns:
        df = df.rename(columns={"eid": "f.eid"})
        
    filtered_train = df[df['f.eid'].isin(train_samples)]
    samplelength = len(train_samples)+len(test_samples)

    train_dir = f'/mypath/{foldername}/data_train'
    test_dir = f'/mypath/{foldername}/data_test'
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    filtered_train.to_csv(f'{train_dir}/{df_name}_{samplelength}.tsv', sep='\t', index=False)

    filtered_test = df[df['f.eid'].isin(test_samples)]
    filtered_test.to_csv(f'{test_dir}/{df_name}_{samplelength}.tsv', sep='\t', index=False)


def sample_store(foldername, df_name, df, samples, letter):
    if 'f.eid' not in df.columns:
        df = df.rename(columns={"eid": "f.eid"})
        
    filtered = df[df['f.eid'].isin(samples)]
    samplelength = len(samples)

    ddir = f'/mypath/{foldername}/data_{letter}'
   
    if not os.path.exists(ddir):
        os.makedirs(ddir)
    filtered.to_csv(f'{ddir}/{df_name}_{samplelength}.tsv', sep='\t', index=False)


def find_chapter(df, letter):
    results = []

    # Iterate through each row (patients)
    for index, row in df.iterrows():
        min_value = None
        result_row = None
        # Loop through each feature (skipping eID)
        for col_idx, value in enumerate(row[1:], start=1):
            if letter in str(value):
                # Calculate the corresponding column index (col_idx + 259)
                new_col_idx = col_idx + int((len(df.columns)-1)/2)
                # Ensure the column index is within bounds
                if new_col_idx < len(df.columns):
                    # Get the corresponding date of the found disease classification
                    #corresponding_value = row[new_col_idx]
                    corresponding_value = row.iloc[new_col_idx]
                    # Check if this is the first found value or if it has the lowest corresponding value
                    # ASSUMPTION: take fist diagnosis of the ICD10 codes in the list
                    if min_value is None or corresponding_value < min_value:
                        min_value = corresponding_value
                        result_row = [row['f.eid'], value, corresponding_value]
        
        # If a result_row was found, append it to the results
        if result_row:
            results.append(result_row)
    # Create a new DataFrame from the results
    result_df = pd.DataFrame(results, columns=['f.eid', 'ICD10', 'date'])
    return result_df


def find_disease(df, list):
    results = []

    # Iterate through each row (patients)
    for index, row in df.iterrows():
        min_value = None
        result_row = None
        
        # Loop through each feature (skipping eID)
        for col_idx, value in enumerate(row[1:], start=1):
            if value in list:
                # Calculate the corresponding column index (col_idx + 259)
                new_col_idx = col_idx + int((len(df.columns)-1)/2)
                
                # Ensure the column index is within bounds
                if new_col_idx < len(df.columns):
                    # Get the corresponding date of the found disease classification
                    #corresponding_value = row[new_col_idx]
                    corresponding_value = row.iloc[new_col_idx]
                    
                    # Check if this is the first found value or if it has the lowest corresponding value
                    # ASSUMPTION: take fist diagnosis of the ICD10 codes in the list
                    if min_value is None or corresponding_value < min_value:
                        min_value = corresponding_value
                        result_row = [row['f.eid'], value, corresponding_value]
        
        # If a result_row was found, append it to the results
        if result_row:
            results.append(result_row)

    # Create a new DataFrame from the results
    result_df = pd.DataFrame(results, columns=['f.eid', 'ICD10', 'date'])
    return result_df


def find_sr_disease(df, list):
    results = []

    # Iterate through each row (patients)
    for index, row in df.iterrows():
        min_value = None
        result_row = None
        
        # Loop through each feature (skipping eID)
        for col_idx, value in enumerate(row[1:], start=1):
            if value in list:
               result_row = [row['f.eid'], value]
        
        # If a result_row was found, append it to the results
        if result_row:
            results.append(result_row)

    # Create a new DataFrame from the results
    result_df = pd.DataFrame(results, columns=['f.eid', 'self_reporeted_bl'])
    return result_df


def load_pt(path, name):
    '''
    Function that loads the encoded data (stored as .pt files) and returns the encoded data as df, the feature names and mapping
    '''
    encoded = torch.load(f'{path}/{name}.pt')
    shape = encoded['tensor'].shape
    try:
        encoded_df = pd.DataFrame(encoded['tensor'].detach().numpy(), columns=[i+f'_{name}' for i in encoded['feature_names']])
        feature_names = encoded['feature_names']
        mapping = None
    except:
        try:
            encoded_df = pd.DataFrame(encoded['tensor'].reshape(encoded['tensor'].size()[0], -1).detach().numpy(), columns=[encoded['feature_names'][0]+f'_{i}' for i in encoded['mapping'].keys()])
            feature_names = encoded['feature_names']
            mapping = encoded['mapping']
        except:
            encoded_df = pd.DataFrame(encoded['tensor'].reshape(encoded['tensor'].size()[0], -1).detach().numpy(), columns=[encoded['feature_names'][j]+f'_{i}' for j in range(len(encoded['feature_names'])) for i in encoded['mapping'].keys()])
            feature_names = encoded['feature_names']
            mapping = encoded['mapping']
    return encoded_df, feature_names, mapping, shape


def write_pt(path, name, encoded_df, residuals, fn, map, normalize=False, shape=None):
    print(name, encoded_df, fn)
    '''
    Given a path, name, df, residuals, feature names an eventually a mapping,
    save the information like the encoded data was moved -> creates a .pt file with the same structure but residualized data.
    '''
    res_df = residuals[encoded_df.columns]
    
    if normalize is True:
        res_df = pd.DataFrame(move.data.preprocessing.standardize(res_df), columns=res_df.columns)

    if map is not None:
        res_dict = {'dataset_name': name, 'tensor': torch.tensor(np.reshape(res_df.to_numpy(dtype='float32'), shape)), 'feature_names': fn, 'mapping': map}
    else:
        res_dict = {'dataset_name': name, 'tensor': torch.tensor(np.reshape(res_df.to_numpy(dtype='float32'), shape)), 'feature_names': fn}
    print(res_dict)
    torch.save(obj=res_dict, f=f'{path}/{name}.pt')


def data_yaml_from_template(temp_path, size, run, letter, interim_run=True, result_run=True):

    res = temp_path.split('_template.yaml')[0].split('_')[-1]

    if interim_run == 'True' or result_run == 'True':
        new_path = temp_path.split('template.yaml')[0] + f'{size}_{run}.yaml'
        
    else:
        new_path = temp_path.split('template.yaml')[0] + f'{size}_{letter}.yaml'

    with open(temp_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    yaml_data['raw_data_path'] = temp_path.split('config/')[0] + f'data_{letter}'

    if interim_run == 'True':
        yaml_data['interim_data_path'] = temp_path.split('config/')[0] + f'interim_data_{res}_{size}_{run}'
    else:
        yaml_data['interim_data_path'] = temp_path.split('config/')[0] + f'interim_data_{res}_{size}_{letter}'
    if result_run == 'True':
        yaml_data['results_path'] = temp_path.split('config/')[0] + f'results_{res}_{size}_{run}'
    else:
        yaml_data['results_path'] = temp_path.split('config/')[0] + f'results_{res}_{size}_{letter}'
    yaml_data['sample_names'] = f'id_{size}'

    if 'categorical_inputs' in yaml_data:
        for item in yaml_data['categorical_inputs']:
            item['name'] = item['name'] + f'_{size}'
    if 'continuous_inputs' in yaml_data:
        for item in yaml_data['continuous_inputs']:
            item['name'] = item['name'] + f'_{size}'

    with open(new_path, 'w') as file:
        yaml.dump(yaml_data, file, default_flow_style=False)
    return


if __name__=='__main__':
    pass