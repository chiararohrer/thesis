import numpy as np
import pandas as pd
import torch
import yaml
import shap
import xgboost
import pickle
import sys



def run_shap(mypath, folder, result_folder, id_file, upper_cutoff, letter, interim_folder):
    sys.path.insert(1, f'{mypath}/python_scripts')
    from wrapper import prepare_latent

    size = int(id_file.split('_')[-1])
    res = interim_folder.split('_')[2]
    with open(f'{mypath}/diseases/icd10_diseases_v2.yaml', 'r') as file:
        icd10_diseases = yaml.safe_load(file)
    disease_names = list(icd10_diseases.keys())

    folder, result_folder, id_file, upper_cutoff, letter = 'prep3', 'results_res_400000_A', 'id_400000', 3650, 'A'
    interim_folder = 'interim_data_res_400000_A'

    interim_path = f'{mypath}/{folder}/{interim_folder}'
    id_path = f'{mypath}/{folder}/data_{letter}/{id_file}.txt'
    indices = torch.load(f'{interim_path}/indices.pt')
    train, test = indices['train_indices'], indices['test_indices']
    eid = pd.read_csv(id_path, sep=' ', header=None, names=['f.eid'])
    train_id, test_id = eid.iloc[train], eid.iloc[test]

    for m in ['cph', 'aft']:

        # XGB Models
        shap_df = pd.DataFrame(columns=['Condition']+[f'dim{i}' for i in range(0,128)])
        for d, disease_name in enumerate(disease_names):

            print(f'starting {disease_name}...')
            latent_full, latent_tsne, latent_dim = prepare_latent(disease_name, folder, result_folder, id_file, upper_cutoff, letter)
            latent_full_train, latent_full_test = latent_full[latent_full['f.eid'].isin(train_id['f.eid'].tolist())], latent_full[latent_full['f.eid'].isin(test_id['f.eid'].tolist())]
            loaded_model = xgboost.Booster()
            loaded_model.load_model(f'{mypath}/baseline/{folder}_{res}_{size}_{letter}0/model_latent_xgb{m}_{folder}_{res}_{size}_{letter}0_{disease_name}_{upper_cutoff}.json')
            explainer = shap.TreeExplainer(loaded_model)
            shap_values = explainer.shap_values(latent_full_train.iloc[:, [i for i in range(1,latent_dim+1)]].to_numpy())
            abs_mean = abs(shap_values).mean(axis=0)
            shap_df.loc[d] = [disease_name]+abs_mean.tolist()

        shap_df=shap_df.set_index('Condition')
        shap_df.to_csv(f'{mypath}/baseline/{folder}_{res}_{size}_{letter}0/latent_shap_xgb{m}.tsv', sep='\t')



        # linear models
        shap_df = pd.DataFrame(columns=['Condition']+[f'dim{i}' for i in range(0,128)])
        pval_df = pd.DataFrame(columns=['Condition']+[f'dim{i}' for i in range(0,128)])
        for d, disease_name in enumerate(disease_names):
            print(f'starting {disease_name}...')
            latent_full, latent_tsne, latent_dim = prepare_latent(disease_name, folder, result_folder, id_file, upper_cutoff, letter)
            latent_full_train, latent_full_test = latent_full[latent_full['f.eid'].isin(train_id['f.eid'].tolist())], latent_full[latent_full['f.eid'].isin(test_id['f.eid'].tolist())]
            try:
                with open(f'{mypath}/baseline/{folder}_{res}_{size}_{letter}0/model_latent_{m}_{folder}_{res}_{size}_{letter}0_{disease_name}_{upper_cutoff}.pkl', 'rb') as file:
                    loaded_model = pickle.load(file)
            
                inter = loaded_model.summary[loaded_model.summary.index.get_level_values('covariate') != 'Intercept'].reset_index()
                inter['dim'] = [int(inter['covariate'][i].split('im')[-1]) for i in range(128)]
                inter = inter.sort_values(by=['dim'])

                shaps = abs(inter['coef']).to_numpy()
                shap_df.loc[d] = [disease_name]+shaps.tolist()

                pvals = (inter['-log2(p)']).to_numpy()
                pval_df.loc[d] = [disease_name]+pvals.tolist()
            except:
                pass
        shap_df=shap_df.set_index('Condition')
        pval_df = pval_df.set_index('Condition')

        shap_df.to_csv(f'{mypath}/baseline/{folder}_{res}_{size}_{letter}0/latent_coef_{m}.tsv', sep='\t')
        pval_df.to_csv(f'{mypath}/baseline/{folder}_{res}_{size}_{letter}0/latent_pval_{m}.tsv', sep='\t')



if __name__ == "__main__":
    mypath = sys.argv[1]
    folder = sys.argv[2]
    result_folder = sys.argv[3]
    id_file = sys.argv[4]
    upper_cutoff = int(sys.argv[5])
    letter = sys.argv[6]
    interim_folder = 'interim_data_' + result_folder.split('_')[1] + '_' + result_folder.split('_')[2] + '_' + result_folder.split('_')[3]
    
    # e.g. python disease_shap_analysis.py "/mypath" "prep3" "results_res_400000_A" "id_400000" "3650" "A"
    
    run_shap(mypath, folder, result_folder, id_file, upper_cutoff, letter, interim_folder)