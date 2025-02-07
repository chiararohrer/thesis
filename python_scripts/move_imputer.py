import numpy as np
import yaml
import torch
from captum.attr import IntegratedGradients
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
from move.models.base import reload_vae
from move.analysis.metrics import calculate_cosine_similarity

from torch.utils.data import DataLoader, TensorDataset
import shap
import seaborn as sns
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from tqdm.notebook import trange, tqdm
import random
from matplotlib.lines import Line2D


def random_zero_num(df, num_to_zero=1000):
    new_df = df.copy()
    temp_df = df.copy()
    mask = new_df != df
    
    for col in new_df.columns[1:]:  # Skip the first column
        print('COL: ', col)
        if 'chr' in col:
            new_df[col] = new_df[col].replace(0, -1)
            temp_df[col] = temp_df[col].replace(0, -1)

        # Get indices of non-zero values in the column
        non_zero_indices = new_df.index[new_df[col] != 0]
        
        # Check if there are enough non-zero values to set
        if len(non_zero_indices) > num_to_zero:
            # Randomly select indices to set to zero
            zero_indices = np.random.choice(non_zero_indices, num_to_zero, replace=False)
            new_df.loc[zero_indices, col] = 0
        else:
            print('less than requested numbers of values detected', col)
            # If fewer than num_to_zero, set all non-zero values to zero
            new_df.loc[non_zero_indices, col] = 0
        mask[col] = new_df[col] != temp_df[col]
        
        if 'chr' in col:
            new_df[col] = new_df[col].replace(-1, 0)

    return new_df, mask

random.seed(6)
np.random.seed(6)

folder = 'prep3'

interim_data_path = f'/mypath/{folder}/interim_data_res_52026_T'
config_path = f"/mypath/{folder}/config/data/{folder}_res_52026_T.yaml"
model_path = f"/mypath/{folder}/results_res_400000_A/latent_space/model.pt"



with open(config_path, "r") as file:
    config = yaml.safe_load(file)


cont, cat = [], []
for item in config['continuous_inputs']:
    cont = cont+[item['name']]
for item in config['categorical_inputs']:
    cat = cat+[item['name']]

ids = pd.read_csv(f'/mypath/{folder}/data_T/id_52026.txt', sep=' ', header=None, names=['f.eid'])

# reload pretrained model and print model info
model = reload_vae(model_path)
model.eval()


all_names = []
categories = []
big_tensor = None
for name in cat+cont:
    dataset = torch.load(f'{interim_data_path}/{name}.pt')

    if big_tensor is not None:
        if len(dataset['tensor'].shape)<3:
            big_tensor = torch.cat((big_tensor, dataset['tensor']), 1)
            all_names = all_names + dataset['feature_names'] 
            categories = categories + [[name] * len(dataset['feature_names'])]
        else:
            big_tensor = torch.cat((big_tensor, dataset['tensor'].reshape(dataset['tensor'].size()[0], -1)), 1)
        
            col_new = [f'{i}_{j}' for i in dataset['feature_names'] for j in range(0,dataset['tensor'].shape[-1])]
            all_names = all_names + col_new
            categories = categories + [[name] * len(col_new)]
    else:
        big_tensor = dataset['tensor'].reshape(dataset['tensor'].size()[0], -1)
        
        col_new = [f'{i}_{j}' for i in dataset['feature_names'] for j in range(0,dataset['tensor'].shape[-1])]
        all_names = all_names + col_new
        categories = categories + [[name] * len(col_new)]

# Create a mask where rows with no values above 1000 are kept
xmask = (big_tensor <= 1000).all(dim=1)
# Apply the mask to keep only the rows that satisfy the condition

big_tensor = big_tensor[xmask, :]

ids = ids[xmask.detach().numpy()]


categories = [item for sublist in categories for item in sublist]
big_tensor_loop = big_tensor
big_tensor = big_tensor.detach().numpy()




big_df = pd.DataFrame(data=big_tensor, columns=all_names)


# Drop nans and reset index
big_df = big_df.dropna().reset_index(drop=True)
ids = ids.dropna().reset_index(drop=True)

big_df = ids.join(big_df)
#big_df = pd.concat([ids, big_df], axis=1, ignore_index=True)



if not os.path.isfile(f'/mypath/{folder}/imputation_results/part1.tsv'):

    random_num, drop_mask = random_zero_num(big_df, num_to_zero=2000)
    random_5prc = random_num

    drop_mask = drop_mask[drop_mask.columns[1:]].to_numpy()

    random_5tensor = torch.tensor(random_5prc[random_5prc.columns[1:]].values, dtype=torch.float32)

    # ### Continuing with imputation:
    z, _ = model.encode(random_5tensor)
    recon = model.decode(z)

    all_recon = []
    for i in range(10):
        all_recon += [model.forward(random_5tensor)['x_recon'].detach().numpy()]

    stack_temp = np.stack(all_recon, axis=0)

    mean_array = np.mean(stack_temp, axis=0)

    variance_array = np.std(stack_temp, axis=0)

    cos_sim = calculate_cosine_similarity(big_tensor*drop_mask, recon[0].detach().numpy()*drop_mask)
    avg_cos_sim = calculate_cosine_similarity(big_tensor*drop_mask, mean_array*drop_mask)

    array1, array2 = big_tensor, mean_array

    # Step 1: Identify differing positions
    #mask = array1 != array2  # Boolean mask where values differ
    mask = drop_mask

    # Step 2: For each column, extract differing values and plot correlations
    n_columns = array1.shape[1]

    #plt.figure(figsize=(10, n_columns * 3))  # Dynamically set figure size
    all_pearsons, mse = [], []

    for col in range(n_columns):
        # Extract differing values for the current column
        diff_indices = mask[:, col]  # Rows where values differ
        values1 = array1[diff_indices, col]  # From array1
        values2 = array2[diff_indices, col]  # From array2
        
        if len(values1) > 1 and len(values2) > 1:  # Only plot if there are differing values

            correlation_coefficient, p_value = pearsonr(values1, values2)
            all_pearsons += [correlation_coefficient]
            mse += [mean_squared_error(values1, values2)]
            
        else:
            correlation_coefficient, p_value = np.NaN, np.NaN
            all_pearsons += [correlation_coefficient]
            mse += [np.NaN]

    # Map categories to colors
    unique_categories = list(set(categories))
    #category_colors = {cat: color for cat, color in zip(unique_categories, ['red', 'blue', 'green', 'yellow', 'black', 'orange', 'purple', 'azure'])}
    category_colors = {cat: color for cat, color in zip(unique_categories, sns.color_palette("husl", n_colors=9))}

    # Assign colors to data points based on categories
    colors = [category_colors[cat] for cat in categories]

    completeness = pd.read_csv(f'/mypath/patients/feature_completeness_{folder}.csv')

    info_df = pd.DataFrame(data=np.array([all_names, np.std(mean_array, axis=0).tolist(), all_pearsons]).T, columns=['feature', 'std', 'pearson'])

    merged_df = info_df.merge(completeness, on='feature', how='left')
    merged_df['completeness'] = merged_df['completeness'].fillna(1)

    merged_df[['std', 'pearson', 'completeness']] = merged_df[['std', 'pearson', 'completeness']].astype(float)

    encoder_captum = pd.read_csv(f"/mypath/{folder}/results_res_400000_A/attributes_e_a_5000.csv", header=None, names=all_names)
    
    e_c = encoder_captum.T.sum(axis=1).rename('e_sum')

    merged_df['encoder_sum'] = merged_df['feature'].map(e_c)
    merged_df['dataset'] = [next(key.split('_6')[0] for key, value in category_colors.items() if value == c) for c in colors]

    # SAVE as tsv
    merged_df.to_csv(f'/mypath/{folder}/imputation_results/part1.tsv', sep='\t')


#### PART 2: Remove a whole dataset at once.

pearson_bin, name_bin, mse_bin, mean_pearson_bin, mean_mse_bin = [], [], [], [], []
spearman_bin, mean_spearman_bin = [], []
cos_sim_bin = []
plot_std = []
plot_pear = []
plot_names = []

start_col, end_col = 0,0

for name1 in cat+cont:
    one_missing = None
    all_names = []
    name_bin += [name1]

    for name in cat+cont:
        dataset = torch.load(f'{interim_data_path}/{name}.pt')
        if name1==name:
            
            if len(dataset['tensor'].shape)<3:
                dataset['tensor'] = dataset['tensor'].zero_()
                plot_names += [dataset['feature_names']]
                start_col = end_col
                end_col = start_col + len(dataset['feature_names'])
            else:
                dataset['tensor'] = dataset['tensor'].zero_()

                col_new = [f'{i}_{j}' for i in dataset['feature_names'] for j in range(0,dataset['tensor'].shape[-1])]
                plot_names += [col_new]

                start_col = end_col
                end_col = start_col + len(col_new)
        
        if len(dataset['tensor'].shape)<3:
            all_names = all_names + dataset['feature_names']
        else:
            col_new = [f'{i}_{j}' for i in dataset['feature_names'] for j in range(0,dataset['tensor'].shape[-1])]
            all_names = all_names + col_new

        if one_missing is not None:
            try:
                one_missing = torch.cat((one_missing, dataset['tensor']), 1)  
            except:
                one_missing = torch.cat((one_missing, dataset['tensor'].reshape(dataset['tensor'].size()[0], -1)), 1)
        else: 
            one_missing = dataset['tensor'].reshape(dataset['tensor'].size()[0], -1)
    
    one_missing = one_missing[xmask, :]
    one_missing = one_missing.numpy()

    one_missing_df = pd.DataFrame(data=one_missing, columns=all_names)
    
    # Drop nans and reset index
    one_missing_df = one_missing_df.dropna().reset_index(drop=True)
    ids = ids.dropna().reset_index(drop=True)

    one_missing_df = ids.join(one_missing_df)

    drop_mask = one_missing_df != big_df
    print('drop mask nan sum', drop_mask.isna().sum())
    drop_mask = drop_mask[drop_mask.columns[1:]].to_numpy()
    one_missing_tensor = torch.tensor(one_missing_df[one_missing_df.columns[1:]].values, dtype=torch.float32)
    
    z, _ = model.encode(one_missing_tensor)
    recon = model.decode(z)

    all_recon = []
    for i in range(10):
        recon = model.forward(one_missing_tensor)['x_recon'].detach().numpy()
        all_recon += [recon]

    mean_array = np.mean(np.stack(all_recon, axis=0), axis=0)

    big_tensor=big_tensor[~np.isnan(big_tensor).any(axis=1)]

    array1, array2 = big_tensor, mean_array

    mask = drop_mask

    n_columns = array1.shape[1]
    all_pearsons, mse = [], []
    mean_pearsons, mean_mse = [], []
    all_spearman, mean_spearman = [], []

    cos_sim = calculate_cosine_similarity(big_tensor*mask, mean_array*mask)
    cos_sim_bin += [cos_sim]

    for col in range(n_columns):
        # Extract differing values for the current column
        diff_indices = mask[:, col]  # Rows where values differ
        
        values1 = array1[diff_indices, col]  # From array1
        values2 = array2[diff_indices, col]  # From array2
        
        if len(values1) > 0 and len(values2) > 0:  # Only plot if there are differing values
            # OBS: take all values in the column for correlation:
            if 'genomics' in name1 or 'pQTL' in name1:
                values1 = array1[:, col]  # From array1
                values2 = array2[:, col]
            try:
                correlation_coefficient, p_value = pearsonr(values1, values2)
                all_pearsons += [correlation_coefficient]
                correlation_coefficient, p_value = spearmanr(values1, values2)
                all_spearman += [correlation_coefficient]
            except:
                correlation_coefficient, p_value = np.NaN, np.NaN
                all_pearsons += [correlation_coefficient]
                all_spearman += [correlation_coefficient]
            try:
                mse += [mean_squared_error(values1, values2)]
            except:
                mse += [np.Nan]
            
            mean_pearsons += [correlation_coefficient]
            mean_spearman += [correlation_coefficient]
            mean_mse += [mean_squared_error(values1, np.zeros(values1.shape))]
        else:
            correlation_coefficient, p_value = np.NaN, np.NaN
            all_pearsons += [correlation_coefficient]
            pass

    nanstd = np.nanstd(mean_array*mask, axis=0)
    nanstd = [nv for nv in nanstd[start_col:end_col]]

    plot_std += [nanstd]
    plot_pear += [all_pearsons[start_col:end_col]]
            
    pearson_bin += [all_pearsons[start_col:end_col]]
    mse_bin += [mse]
    mean_pearson_bin += [mean_pearsons]
    mean_mse_bin += [mean_mse]
    spearman_bin += [all_spearman]
    mean_spearman_bin += [mean_spearman]

plot_std_flat = [item for sublist in plot_std for item in sublist]
plot_pear_flat = [item for sublist in plot_pear for item in sublist]
plot_names_flat = [item for sublist in plot_names for item in sublist]

completeness = pd.read_csv(f'/mypath/patients/feature_completeness_{folder}.csv')

encoder_captum = pd.read_csv(f"/mypath/{folder}/results_res_400000_A/attributes_e_a_5000.csv", header=None, names=all_names)
e_c = encoder_captum.T.sum(axis=1).rename('e_sum')

info_df = pd.DataFrame(data=np.array([plot_names_flat, plot_std_flat, plot_pear_flat]).T, columns=['feature', 'std', 'pearson'])

merged_df = info_df.merge(completeness, on='feature', how='left')
merged_df['completeness'] = merged_df['completeness'].fillna(1)
merged_df[['std', 'pearson', 'completeness']] = merged_df[['std', 'pearson', 'completeness']].astype(float)

# SAVE to tsv
merged_df.to_csv(f'/mypath/{folder}/imputation_results/part2_v3.tsv', sep='\t')