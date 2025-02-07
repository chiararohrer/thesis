from sklearn.model_selection import train_test_split
import torch
import os
import sys
from sklearn.linear_model import LogisticRegression
import shap
import yaml
import sksurv
import pickle
import time
import numpy as np
import pandas as pd
import xgboost
from lifelines import CoxPHFitter, LogNormalAFTFitter
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tabulate import tabulate
from SecretColors import Palette
import matplotlib as mpl
import random
import datetime

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=Palette('material', seed=10).random(no_of_colors=20, shade=60))


def run_all(folder, result_folder, interim_folder, id_file, output_folder, upper_cutoff=3650, letter=None, store_models=True, seed=None):
    print('in run all: ', folder, result_folder, interim_folder, id_file, output_folder, upper_cutoff)

    if seed is not None:
        random.seed(seed)

    os.makedirs(output_folder, exist_ok=True)

    model_names = ['Condition']+[f'Model {x}' for x in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L']]
    result_matrix = pd.DataFrame(columns=model_names)
    
    with open('/mypath/diseases/icd10_diseases_v2.yaml', 'r') as file:
        icd10_diseases = yaml.safe_load(file)

    disease_names = list(icd10_diseases.keys())

    for disease in disease_names:
        print('starting ', disease)
        ci_a, ci_b, ci_c, ci_d, ci_e, ci_f, ci_g, ci_h, ci_k, ci_l = predict_evaluate(disease, folder, result_folder, interim_folder, id_file, output_folder, upper_cutoff, letter, store_models, seed)
        row = [disease, ci_a, ci_b, ci_c, ci_d, ci_e, ci_f, ci_g, ci_h, ci_k, ci_l]
        result_matrix = result_matrix._append(pd.Series(row, index=model_names), ignore_index=True)
    
    rename_dict = {
        'Model A': 'latent_cph',
        'Model B': 'lasso_cph',
        'Model C': 'full_xgbcph',
        'Model D': 'latent_xgbcph',
        'Model E': 'lasso_xgbcph',
        'Model F': 'full_xgbaft',
        'Model G': 'latent_xgbaft',
        'Model H': 'lasso_xgbaft',
        'Model K': 'latent_aft',
        'Model L': 'lasso_aft',
    }

    result_matrix = result_matrix.rename(columns=rename_dict, inplace=False)

    print(tabulate(result_matrix, headers='keys', tablefmt='psql', showindex=False))
    output_split = output_folder.split('/')[-1]
    result_matrix.to_csv(f'{output_folder}/results_{output_split}_{upper_cutoff}.tsv', sep='\t', index=True, index_label=False)
        
    results = pd.read_csv(f'{output_folder}/results_{output_split}_{upper_cutoff}.tsv', sep='\t', index_col=0)
    results = results.set_index('Condition')
    ### make new pdf file
    with PdfPages(f'{output_folder}/results_{output_split}_{upper_cutoff}.pdf') as pdf:

        fig, ax = plt.subplots(layout='constrained', figsize=(12,12))
        x = np.arange(len(results.columns.to_list()))
        for i, disease in enumerate(results.index.to_list()):
            group = ax.bar(x + i*0.05, results.loc[disease,:], width =0.05)
        ax.set_xticks(x + 0.3, results.columns.to_list())
        plt.ylabel('C-index')
        ax.legend(results.index.to_list(),loc='upper left', ncols=3)
        pdf.savefig(bbox_inches='tight')
        plt.close()

        idx = {'latent': ['latent_cph', 'latent_xgbcph', 'latent_xgbaft', 'latent_aft'], 'lasso-selected': ['lasso_cph', 'lasso_xgbcph', 'lasso_xgbaft', 'lasso_aft'], 'interim': ['full_xgbcph', 'full_xgbaft']}
        groupnames = list(idx.keys())
        mpl.rcParams['axes.prop_cycle'] = mpl.rcParams['axes.prop_cycle'][2:]

        for i, disease in enumerate(results.index.to_list()):
            fig, ax = plt.subplots(layout='constrained')
            x = [0,0.1,0.2, 0.3]
            x = np.arange(4)
            tick_list, pos_list = [], []
            for j, cat in enumerate(['latent', 'lasso-selected', 'interim']):
                group = ax.bar(x[:len(idx[f'{cat}'])]+5*j, results.loc[disease,idx[f'{cat}']].values, width =0.5)
                tick_list = tick_list + idx[f'{cat}']
                pos_list = pos_list + [x[:len(idx[f'{cat}'])]+5*j]
            pos_list = np.concatenate(pos_list)
            plt.xticks(rotation=45)
            ax.set_xticks(pos_list, tick_list)
            ax.set_ylim(0.5,1)
            ax.legend(groupnames,loc='upper left', ncols=3)
            plt.ylabel('C-index')
            plt.title(f'{disease}')
            pdf.savefig(bbox_inches='tight')
            plt.close()

    return

def run_reduced(folder, result_folder, interim_folder, id_file, output_folder, upper_cutoff=3650, letter=None, store_models=True, seed=None):
    print('in run reduced: ', folder, result_folder, interim_folder, id_file, output_folder, upper_cutoff)

    if seed is not None:
        random.seed(seed)

    os.makedirs(output_folder, exist_ok=True)
    output_split = output_folder.split('/')[-1]

    model_names = ['Condition']+[f'Model {x}' for x in ['A', 'D', 'G', 'K']]
    result_matrix = pd.DataFrame(columns=model_names)

    disease_names = ['type 1 diabetes mellitus', 'type 2 diabetes mellitus', 'primary hypertension', 'renal failure', 'copd']

    for disease in disease_names:
        print('starting ', disease)
        ci_a, ci_d, ci_g, ci_k = predict_evaluate_reduced(disease, folder, result_folder, interim_folder, id_file, output_folder, upper_cutoff, letter, store_models, seed)
        row = [disease, ci_a, ci_d, ci_g, ci_k]
        result_matrix = result_matrix._append(pd.Series(row, index=model_names), ignore_index=True)
    
    rename_dict = {
        'Model A': 'latent_cph',
        'Model D': 'latent_xgbcph',
        'Model G': 'latent_xgbaft',
        'Model K': 'latent_aft',
    }

    result_matrix = result_matrix.rename(columns=rename_dict, inplace=False)

    print(tabulate(result_matrix, headers='keys', tablefmt='psql', showindex=False))
    result_matrix.to_csv(f'{output_folder}/results_{output_split}_{upper_cutoff}.tsv', sep='\t', index=True, index_label=False)
        
    results = pd.read_csv(f'{output_folder}/results_{output_split}_{upper_cutoff}.tsv', sep='\t', index_col=0)
    results = results.set_index('Condition')
    ### make new pdf file
    with PdfPages(f'{output_folder}/results_reduced_{output_split}_{upper_cutoff}.pdf') as pdf:

        fig, ax = plt.subplots(layout='constrained', figsize=(12,12))
        x = np.arange(len(results.columns.to_list()))
        for i, disease in enumerate(results.index.to_list()):
            group = ax.bar(x + i*0.05, results.loc[disease,:], width =0.05)
        ax.set_xticks(x + 0.3, results.columns.to_list())
        plt.ylabel('C-index')
        ax.legend(results.index.to_list(),loc='upper left', ncols=3)
        pdf.savefig(bbox_inches='tight')
        plt.close()

        idx = {'latent': ['latent_cph', 'latent_xgbcph', 'latent_xgbaft', 'latent_aft']}
        groupnames = list(idx.keys())
        mpl.rcParams['axes.prop_cycle'] = mpl.rcParams['axes.prop_cycle'][2:]

        for i, disease in enumerate(results.index.to_list()):
            fig, ax = plt.subplots(layout='constrained')
            x = [0,0.1,0.2, 0.3]
            x = np.arange(4)
            tick_list, pos_list = [], []
            for j, cat in enumerate(['latent']):
                group = ax.bar(x[:len(idx[f'{cat}'])]+5*j, results.loc[disease,idx[f'{cat}']].values, width =0.5)
                tick_list = tick_list + idx[f'{cat}']
                pos_list = pos_list + [x[:len(idx[f'{cat}'])]+5*j]
            pos_list = np.concatenate(pos_list)
            plt.xticks(rotation=45)
            ax.set_xticks(pos_list, tick_list)
            ax.set_ylim(0.5,1)
            ax.legend(groupnames,loc='upper left', ncols=1)
            plt.ylabel('C-index')
            plt.title(f'{disease}')
            pdf.savefig(bbox_inches='tight')
            plt.close()
    return


def predict_evaluate(disease_name, folder, result_folder, interim_folder, id_file, output_folder, upper_cutoff, letter, store_models=True, seed=None):
    os.makedirs(output_folder, exist_ok=True)
    output_split = output_folder.split('/')[-1]

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    cox_params = {"eta": 0.003, "max_depth": 4, "objective": "survival:cox", "subsample": 0.8}
    aft_params = {'objective': 'survival:aft',
            'eval_metric': 'aft-nloglik',
            'aft_loss_distribution': 'normal',
            'aft_loss_distribution_scale': 2,
            'tree_method': 'hist', 'learning_rate': 0.003, 'max_depth': 4, "subsample": 0.8}
    
    print('in predict_evaluate', flush=True)
    # Prepare latent data
    latent_full, latent_tsne, latent_dim = prepare_latent(disease_name, folder, result_folder, id_file, upper_cutoff, letter)
    print('latent data loaded', datetime.datetime.now(), flush=True)

    # Prepare interim data
    interim_path = f'/mypath/{folder}/{interim_folder}'
    try:
        id_path = f'/mypath/{folder}/data_{letter}/{id_file}.txt'
        interim_full_train, interim_full_test, merged_cols, train_id, test_id = load_interim_new(interim_path, id_path)
    except:
        id_path = f'/mypath/{folder}/data/{id_file}.txt'
        interim_full_train, interim_full_test, merged_cols, train_id, test_id = load_interim_new(interim_path, id_path)
    interim_dim = interim_full_train.shape[1]
    interim_full_train, interim_full_test = pd.DataFrame(data=interim_full_train, columns=merged_cols), pd.DataFrame(data=interim_full_test, columns=merged_cols)
    interim_full_train, interim_full_test = pd.concat([train_id.reset_index(drop=True), interim_full_train], ignore_index=False, axis=1), pd.concat([test_id.reset_index(drop=True), interim_full_test], ignore_index=False, axis=1)
    interim_full_train, interim_full_test = prepare_interim(disease_name, folder, result_folder, id_file, upper_cutoff, interim_full_train, interim_full_test)
    print('full data loaded', flush=True)
    if interim_dim > 5000:
        block_full = True
    else:
        block_full = False


    # Split latent data into train and test:
    latent_full_train, latent_full_test = latent_full[latent_full['f.eid'].isin(train_id['f.eid'].tolist())], latent_full[latent_full['f.eid'].isin(test_id['f.eid'].tolist())]
    latent_tsne_train, latent_tsne_test = latent_tsne[latent_tsne['f.eid'].isin(train_id['f.eid'].tolist())], latent_tsne[latent_tsne['f.eid'].isin(test_id['f.eid'].tolist())]

    # Set all times to 0 in case model fitting fails
    time_a, time_b, time_c, time_d, time_e, time_f, time_g, time_h, time_k, time_l = 0,0,0,0,0,0,0,0,0,0
    ci_a, ci_b, ci_c, ci_d, ci_e, ci_f, ci_g, ci_h, ci_k, ci_l = 0,0,0,0,0,0,0,0,0,0

    # Print number of cases:
    print('NUMBER OF POSITIVES (latent): ', latent_full_train['binary'].sum(), latent_full_test['binary'].sum())
    print('NUMBER OF POSITIVES (interim): ', interim_full_train['binary'].sum(), interim_full_test['binary'].sum())
    print('NUMBER OF ALL (interim): ', interim_full_train['binary'].count(), interim_full_test['binary'].count())

    # Model a) CoxPH on latent
    print('----- MODEL A: CoxPH on latent -----', datetime.datetime.now(), flush=True)
    model_a = CoxPHFitter()
    ll_cols = [i for i in range(1,latent_dim+1)]+[-2,-1]
    start_time = time.time()
    try:
        model_a.fit(latent_full_train.iloc[:,ll_cols], 'delta_days', 'binary',show_progress=True, fit_options={'step_size': 0.1})
        end_time = time.time()
        time_a = end_time - start_time
        #model_a.check_assumptions(latent_full_train.iloc[:,ll_cols])
        #model_a.print_summary()
        ci_a = concordance_index_censored(latent_full_test['binary'].astype(bool), latent_full_test['delta_days'], model_a.predict_partial_hazard(latent_full_test.iloc[:,ll_cols]))[0]
        print('----- MODEL A: C-index on test data: ', ci_a, ' -----')
        if store_models == True:
            pickle_path = f'{output_folder}/model_latent_cph_{output_split}_{disease_name}_{upper_cutoff}.pkl'
            with open(pickle_path, 'wb') as file:
                pickle.dump(model_a, file)
    except:
        #ci_a = 0
        print('----- MODEL A: FITTING FAILED -----')


    # Model b) CoxPH on lasso-selected features
    print('----- MODEL B: CoxPH on lasso-selected features -----', datetime.datetime.now(), flush=True)
    lasso = LogisticRegression(penalty='l1', C=0.01, solver='saga', random_state=seed, tol=0.01, max_iter=50)
    X_train = interim_full_train.iloc[:, 1:interim_dim+1]
    E_train = interim_full_train['binary']
    
    start_time = time.time()
    lasso.fit(X_train, E_train)  

    X_test = interim_full_test.iloc[:, 1:interim_dim+1]
    E_test = interim_full_test['binary']
    print(f'score of Lasso on test set: {lasso.score(X_test, E_test)}')

    coefficients = lasso.coef_[0]
    feature_names = X_train.columns
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })
    top_features = feature_importance.reindex(feature_importance.Coefficient.abs().sort_values(ascending=False).index)
    # Get all featuers >0
    top_features = top_features[top_features['Coefficient'].abs()>0]
    try:
        top_features = top_features[:50]
        print('TOP FEATURES CLIPPED AT TOP 50!')
    except:
        pass
    print(top_features, datetime.datetime.now(), flush=True)

    model_b = CoxPHFitter()
    feature_list = top_features['Feature'].to_list()
    block=True
    while block is True and len(feature_list)>0:
        selected_cols = feature_list+['delta_days', 'binary']      
        try:
            model_b.fit(interim_full_train[selected_cols], 'delta_days', 'binary',show_progress=True, fit_options={'step_size': 0.1})
            end_time = time.time()
            time_b = end_time - start_time
            block = False
            print('Selected features: ', len(feature_list), feature_list)
        except:
            feature_list.pop()
            block = True
    #model_b.check_assumptions(interim_full_train[selected_cols])
    #model_b.print_summary()
    
    if len(feature_list)>0:
        ci_b = concordance_index_censored(interim_full_test['binary'].astype(bool), interim_full_test['delta_days'], model_b.predict_partial_hazard(interim_full_test[selected_cols]))[0]
        print('----- MODEL B: C-index on test data: ', ci_b, ' -----')
        if store_models == True:
            pickle_path = f'{output_folder}/model_lasso_cph_{output_split}_{disease_name}_{upper_cutoff}.pkl'
            with open(pickle_path, 'wb') as file:
                pickle.dump(model_b, file)
    else:
        #ci_b = 0
        print('----- MODEL B: Ran out of features -----')
    
    print('----- MODEL C: XGBoost with CoxPH on all features -----', datetime.datetime.now(), flush=True)
    if block_full==False:
        # Model c) XGBoost with CoxPH on all features
        y_train, y_test = [r['delta_days'] if r['binary']==1 else -r['delta_days'] for i, r in interim_full_train.iterrows()], [r['delta_days'] if r['binary']==1 else -r['delta_days'] for i, r in interim_full_test.iterrows()]
        x_train, x_test = interim_full_train.iloc[:, [i for i in range(1,interim_dim+1)]].to_numpy(), interim_full_test.iloc[:, [i for i in range(1,interim_dim+1)]].to_numpy()
        xgb_train = xgboost.DMatrix(x_train, label=y_train)
        xgb_test = xgboost.DMatrix(x_test, label=y_test)
        
        start_time = time.time()
        model_c = xgboost.train(cox_params, xgb_train, 5000, evals=[(xgb_train, "train")], verbose_eval=1000)
        end_time = time.time()
        time_c = end_time - start_time
        ci_c = concordance_index_censored(interim_full_test['binary'].astype(bool), interim_full_test['delta_days'], model_c.predict(xgb_test))[0]
        print('----- MODEL C: C-index on test data: ', ci_c, ' -----')
        c_risk_scores = model_c.predict(xgb_test) #for auroc plot
        if store_models == True:
            pickle_path = f'{output_folder}/model_interim_xgbcph_{output_split}_{disease_name}_{upper_cutoff}.json'
            model_c.save_model(pickle_path)
        ## Make time-dependent AUROC plot for 3 XGB-Cox models:
        va_times = np.arange(interim_full_test['delta_days'].min(), upper_cutoff, 7)
        try:
            c_auc, c_mean_auc = cumulative_dynamic_auc(sksurv.util.Surv.from_dataframe('binary', 'delta_days', interim_full_train), sksurv.util.Surv.from_dataframe('binary', 'delta_days', interim_full_test), c_risk_scores, va_times)
        except:
            c_auc, c_mean_auc = None, None
    else:
        ci_c = 0


    # Model D: XGBoost with CoxPH on latent space
    print('----- MODEL D: XGBoost with CoxPH on latent space -----', datetime.datetime.now(), flush=True)
    y_train, y_test = [r['delta_days'] if r['binary']==1 else -r['delta_days'] for i, r in latent_full_train.iterrows()], [r['delta_days'] if r['binary']==1 else -r['delta_days'] for i, r in latent_full_test.iterrows()]
    xgb_train = xgboost.DMatrix(latent_full_train.iloc[:, [i for i in range(1,latent_dim+1)]].to_numpy(), label=y_train)
    xgb_test = xgboost.DMatrix(latent_full_test.iloc[:, [i for i in range(1,latent_dim+1)]].to_numpy(), label=y_test)
    
    start_time = time.time()
    model_d = xgboost.train(cox_params, xgb_train, 5000, evals=[(xgb_train, "train")], verbose_eval=1000)
    end_time = time.time()
    time_d = end_time - start_time
    ci_d = concordance_index_censored(latent_full_test['binary'].astype(bool), latent_full_test['delta_days'], model_d.predict(xgb_test))[0]
    print('----- MODEL D: C-index on test data: ', ci_d, ' -----')
    d_risk_scores = model_d.predict(xgb_test) #for auroc plot
    if store_models == True:
        pickle_path = f'{output_folder}/model_latent_xgbcph_{output_split}_{disease_name}_{upper_cutoff}.json'
        model_d.save_model(pickle_path)
    try:
        d_auc, d_mean_auc = cumulative_dynamic_auc(sksurv.util.Surv.from_dataframe('binary', 'delta_days', latent_full_train), sksurv.util.Surv.from_dataframe('binary', 'delta_days', latent_full_test), d_risk_scores, va_times)
    except:
        d_auc, d_mean_auc = None, None

    # Model E: XGBoost CoxPH model on lasso-selected features
    print('----- MODEL E: XGBoost with CoxPH on lasso-selected features -----', datetime.datetime.now(), flush=True)
    if len(feature_list)>0:
        y_train, y_test = [r['delta_days'] if r['binary']==1 else -r['delta_days'] for i, r in interim_full_train.iterrows()], [r['delta_days'] if r['binary']==1 else -r['delta_days'] for i, r in interim_full_test.iterrows()]
        x_train, x_test = interim_full_train[selected_cols[:-2]].to_numpy(), interim_full_test[selected_cols[:-2]].to_numpy()
        xgb_train = xgboost.DMatrix(x_train, label=y_train)
        xgb_test = xgboost.DMatrix(x_test, label=y_test)
    
    try:
        start_time = time.time()
        model_e = xgboost.train(cox_params, xgb_train, 5000, evals=[(xgb_train, "train")], verbose_eval=1000)
        end_time = time.time()
        time_e = end_time - start_time
        ci_e = concordance_index_censored(interim_full_test['binary'].astype(bool), interim_full_test['delta_days'], model_e.predict(xgb_test))[0]
        print('----- MODEL E: C-index on test data: ', ci_e, ' -----')
        if store_models == True:
            pickle_path = f'{output_folder}/model_lasso_xgbcph_{output_split}_{disease_name}_{upper_cutoff}.json'
            model_e.save_model(pickle_path)
        e_risk_scores = model_e.predict(xgb_test) #for auroc plot
        e_auc, e_mean_auc = cumulative_dynamic_auc(sksurv.util.Surv.from_dataframe('binary', 'delta_days', interim_full_train[selected_cols]), sksurv.util.Surv.from_dataframe('binary', 'delta_days', interim_full_test[selected_cols]), e_risk_scores, va_times)
        
    except:
        #ci_e = 0
        print('----- MODEL E: FITTING FAILED -----')
    
    
    if block_full==False:
        # Model F: XGBoost AFT model on interim
        print('----- MODEL F: XGBoost with AFT on interim -----', datetime.datetime.now(), flush=True)
        y_upper_train, y_upper_test = np.array([+np.inf if r['binary']==0 else r['delta_days'] for i, r in interim_full_train.iterrows()]), np.array([+np.inf if r['binary']==0 else r['delta_days'] for i, r in interim_full_test.iterrows()])
        y_lower_train, y_lower_test = np.array([r['delta_days'] for i, r in interim_full_train.iterrows()]), np.array([r['delta_days'] for i, r in interim_full_test.iterrows()])
        x_train, x_test = interim_full_train.iloc[:, [i for i in range(1,interim_dim+1)]].to_numpy(), interim_full_test.iloc[:, [i for i in range(1,interim_dim+1)]].to_numpy()
        aft_train = xgboost.DMatrix(x_train)
        aft_test = xgboost.DMatrix(x_test)
        aft_train.set_float_info('label_lower_bound', y_lower_train)
        aft_train.set_float_info('label_upper_bound', y_upper_train)
        aft_test.set_float_info('label_lower_bound', y_lower_test)
        aft_test.set_float_info('label_upper_bound', y_upper_test)
        
        start_time = time.time()
        model_f = xgboost.train(aft_params, aft_train, num_boost_round=5000,
                        evals=[(aft_train, 'train')], verbose_eval=1000)
        end_time = time.time()
        time_f = end_time - start_time
        ci_f = concordance_index_censored(interim_full_test['binary'].astype(bool), interim_full_test['delta_days'], -model_f.predict(aft_test))[0]
        print('----- MODEL F: C-index on test data: ', ci_f, ' -----')
        if store_models == True:
            pickle_path = f'{output_folder}/model_interim_xgbaft_{output_split}_{disease_name}_{upper_cutoff}.json'
            model_f.save_model(pickle_path)
    else:
        ci_f = 0

    # Model G: XGBoost AFT model on latent
    print('----- MODEL G: XGBoost with AFT on latent -----', datetime.datetime.now(), flush=True)
    y_upper_train, y_upper_test = np.array([+np.inf if r['binary']==0 else r['delta_days'] for i, r in latent_full_train.iterrows()]), np.array([+np.inf if r['binary']==0 else r['delta_days'] for i, r in latent_full_test.iterrows()])
    y_lower_train, y_lower_test = np.array([r['delta_days'] for i, r in latent_full_train.iterrows()]), np.array([r['delta_days'] for i, r in latent_full_test.iterrows()])
    x_train, x_test = latent_full_train.iloc[:, [i for i in range(1,latent_dim+1)]].to_numpy(), latent_full_test.iloc[:, [i for i in range(1,latent_dim+1)]].to_numpy()
    aft_train = xgboost.DMatrix(x_train)
    aft_test = xgboost.DMatrix(x_test)
    aft_train.set_float_info('label_lower_bound', y_lower_train)
    aft_train.set_float_info('label_upper_bound', y_upper_train)
    aft_test.set_float_info('label_lower_bound', y_lower_test)
    aft_test.set_float_info('label_upper_bound', y_upper_test)
    
    start_time = time.time()
    model_g = xgboost.train(aft_params, aft_train, num_boost_round=5000,
                    evals=[(aft_train, 'train')], verbose_eval=1000)
    end_time = time.time()
    time_g = end_time - start_time
    ci_g = concordance_index_censored(latent_full_test['binary'].astype(bool), latent_full_test['delta_days'], -model_g.predict(aft_test))[0]
    print('----- MODEL G: C-index on test data: ', ci_g, ' -----')
    if store_models == True:
        pickle_path = f'{output_folder}/model_latent_xgbaft_{output_split}_{disease_name}_{upper_cutoff}.json'
        model_g.save_model(pickle_path)

    # Model H: XGBoost AFT model on lasso-selected features
    print('----- MODEL H: XGBoost with AFT on lasso-selected features -----', datetime.datetime.now(), flush=True)
    if len(feature_list)>0:
        y_upper_train, y_upper_test = np.array([+np.inf if r['binary']==0 else r['delta_days'] for i, r in interim_full_train.iterrows()]), np.array([+np.inf if r['binary']==0 else r['delta_days'] for i, r in interim_full_test.iterrows()])
        y_lower_train, y_lower_test = np.array([r['delta_days'] for i, r in interim_full_train.iterrows()]), np.array([r['delta_days'] for i, r in interim_full_test.iterrows()])
        x_train, x_test = interim_full_train[selected_cols[:-2]].to_numpy(), interim_full_test[selected_cols[:-2]].to_numpy()
        aft_train = xgboost.DMatrix(x_train)
        aft_test = xgboost.DMatrix(x_test)
        aft_train.set_float_info('label_lower_bound', y_lower_train)
        aft_train.set_float_info('label_upper_bound', y_upper_train)
        aft_test.set_float_info('label_lower_bound', y_lower_test)
        aft_test.set_float_info('label_upper_bound', y_upper_test)
    
    try:
        start_time = time.time()
        model_h = xgboost.train(aft_params, aft_train, num_boost_round=5000,
                    evals=[(aft_train, 'train')], verbose_eval=1000)
        end_time = time.time()
        time_h = end_time - start_time
        ci_h = concordance_index_censored(interim_full_test['binary'].astype(bool), interim_full_test['delta_days'], -model_h.predict(aft_test))[0]
        print('----- MODEL H: C-index on test data: ', ci_h, ' -----')
        if store_models == True:
            pickle_path = f'{output_folder}/model_lasso_xgbaft_{output_split}_{disease_name}_{upper_cutoff}.json'
            model_h.save_model(pickle_path)
    except:
        #ci_h = 0
        print('----- MODEL H: FITTING FAILED -----')

    # Model K: AFT on latent
    print('----- MODEL K: LogNormalAFTFitter on latent -----', datetime.datetime.now(), flush=True)
    model_k = LogNormalAFTFitter()
    ll_cols = [i for i in range(1,latent_dim+1)]+[-2,-1]
    try: 
        start_time = time.time()
        model_k.fit(latent_full_train.iloc[:,ll_cols], 'delta_days', 'binary',show_progress=True, fit_options={'step_size': 0.1})
        end_time = time.time()
        time_k = end_time - start_time
        
        ci_k = concordance_index_censored(latent_full_test['binary'].astype(bool), latent_full_test['delta_days'], -model_k.predict_expectation(latent_full_test.iloc[:,ll_cols]))[0]
        print('----- MODEL K: C-index on test data: ', ci_k, ' -----')
        if store_models == True:
            pickle_path = f'{output_folder}/model_latent_aft_{output_split}_{disease_name}_{upper_cutoff}.pkl'
            with open(pickle_path, 'wb') as file:
                pickle.dump(model_k, file)
    except: 
        #ci_k = 0
        print('----- MODEL K: FITTING FAILED -----')

    # Model L: AFT normal (lifelines) on lasso-selected
    print('----- MODEL L: LogNormalAFTFitter on lasso -----', datetime.datetime.now(), flush=True)
    model_l = LogNormalAFTFitter()
    try:
        start_time = time.time()
        model_l.fit(interim_full_train[selected_cols], 'delta_days', 'binary',show_progress=True, fit_options={'step_size': 0.1})
        end_time = time.time()
        time_l = end_time - start_time
        
        ci_l = concordance_index_censored(interim_full_test['binary'].astype(bool), interim_full_test['delta_days'], -model_l.predict_expectation(interim_full_test[selected_cols]))[0]
        print('----- MODEL L: C-index on test data: ', ci_l, ' -----')
        if store_models == True:
            pickle_path = f'{output_folder}/model_lasso_aft_{output_split}_{disease_name}_{upper_cutoff}.pkl'
            with open(pickle_path, 'wb') as file:
                pickle.dump(model_l, file)
    except: 
        #ci_l = 0
        print('----- MODEL L: FITTING FAILED -----')


    print('---------- C-INDEX SUMMARY ----------')
    print('MODEL A: ', ci_a, time_a)
    print('MODEL B: ', ci_b, time_b)
    print('MODEL C: ', ci_c, time_c)
    print('MODEL D: ', ci_d, time_d)
    print('MODEL E: ', ci_e, time_e)
    print('MODEL F: ', ci_f, time_f)
    print('MODEL G: ', ci_g, time_g)
    print('MODEL H: ', ci_h, time_h)
    print('MODEL K: ', ci_k, time_k)
    print('MODEL L: ', ci_l, time_l)

    with PdfPages(f'{output_folder}/report_{output_split}_{disease_name}_{upper_cutoff}.pdf') as pdf:

        firstPage = plt.figure(figsize=(8,6))
        firstPage.clf()
        title = f'Survival model results for {disease_name} within {upper_cutoff} days from baseline visit:'
        info = f'From: {folder}, {result_folder}'
        txt = f'\
            Model A: CoxPH on latent: C-index A: {round(ci_a,3)} ; fitting took {time_a} \n\
            Model B: CoxPH on lasso-selected: C-index B: {round(ci_b,3)} ; fitting took {time_b} \n\
            Model C: XGB_Cox on interim: C-index C: {round(ci_c,3)} ; fitting took {time_c} \n\
            Model D: XGB_Cox on latent: C-index D: {round(ci_d,3)} ; fitting took {time_d} \n\
            Model E: XGB_Cox on lasso-selected: C-index E: {round(ci_e,3)} ; fitting took {time_e} \n\
            Model F: XGB_AFT on interim: C-index F: {round(ci_f,3)} ; fitting took {time_f} \n\
            Model G: XGB_AFT on latent: C-index G: {round(ci_g,3)} ; fitting took {time_g} \n\
            Model H: XGB_AFT on lasso-selected: C-index H: {round(ci_h,3)} ; fitting took {time_h} \n\
            Model K: AFT on latent: C-index K: {round(ci_k,3)} ; fitting took {time_k} \n\
            Model L: AFT on lasso-selected: C-index L: {round(ci_l,3)} ; fitting took {time_l} \n\
                '
        firstPage.text(0.05,0.9, title, transform=firstPage.transFigure, size=12, ha='left')
        firstPage.text(0.05,0.8, info, transform=firstPage.transFigure, size=12, ha='left')
        firstPage.text(0.1,0.3,txt, transform=firstPage.transFigure, size=12, ha='left')
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # A
        fig, ax =plt.subplots(figsize=(12,30))
        ax.axis('tight')
        ax.axis('off')
        try:
            ax.table(cellText=model_a.summary.reset_index().values,colLabels=model_a.summary.reset_index().columns,loc='center')
        except:
            pass
        plt.title('Model A: CoxPH on latent')
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # B
        fig, ax =plt.subplots(figsize=(12,10))
        ax.axis('tight')
        ax.axis('off')
        try:
            ax.table(cellText=model_b.summary.reset_index().values,colLabels=model_b.summary.reset_index().columns,loc='center')
        except:
            pass
        plt.title(f'Model B: CoxPH on lasso-selected')
        pdf.savefig(bbox_inches='tight')
        plt.close()
        
        if block_full==False:
            # C
            shap_values = shap.TreeExplainer(model_c).shap_values(interim_full_train.iloc[:, [i for i in range(1,interim_dim+1)]].to_numpy())
            shap.summary_plot(shap_values, interim_full_train.iloc[:, [i for i in range(1,interim_dim+1)]], show=False)
            plt.title('Model C: XGB_Cox on interim')
            pdf.savefig(bbox_inches='tight')
            plt.close()

        # D
        shap_values = shap.TreeExplainer(model_d).shap_values(latent_full_train.iloc[:, [i for i in range(1,latent_dim+1)]].to_numpy())
        shap.summary_plot(shap_values, latent_full_train.iloc[:, [i for i in range(1,latent_dim+1)]], show=False)
        plt.title('Model D: XGB_Cox on latent')
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # E
        try:
            shap_values = shap.TreeExplainer(model_e).shap_values(interim_full_train[selected_cols[:-2]].to_numpy())
            shap.summary_plot(shap_values, interim_full_train[selected_cols[:-2]], show=False)
            plt.title('Model E: XGB_Cox on lasso-selected')
            pdf.savefig(bbox_inches='tight')
            plt.close()
        except:
            pass

        if block_full==False:
            # F
            try:
                shap_values = shap.TreeExplainer(model_f).shap_values(interim_full_train.iloc[:, [i for i in range(1,interim_dim+1)]].to_numpy())
                shap.summary_plot(shap_values, interim_full_train.iloc[:, [i for i in range(1,interim_dim+1)]], show=False)
                plt.title('Model F: XGB_AFT on interim')
                pdf.savefig(bbox_inches='tight')
                plt.close()
            except:
                pass

        # G
        try:
            shap_values = shap.TreeExplainer(model_g).shap_values(latent_full_train.iloc[:, [i for i in range(1,latent_dim+1)]].to_numpy())
            shap.summary_plot(shap_values, latent_full_train.iloc[:, [i for i in range(1,latent_dim+1)]], show=False)
            plt.title('Model G: XGB_AFT on latent')
            pdf.savefig(bbox_inches='tight')
            plt.close()
        except:
            pass

        # H
        try:
            shap_values = shap.TreeExplainer(model_h).shap_values(interim_full_train[selected_cols[:-2]].to_numpy())
            shap.summary_plot(shap_values, interim_full_train[selected_cols[:-2]], show=False)
            plt.title('Model H: XGB_AFT on lasso-selected')
            pdf.savefig(bbox_inches='tight')
            plt.close()
        except:
            pass

        # K
        fig, ax =plt.subplots(figsize=(12,30))
        ax.axis('tight')
        ax.axis('off')
        try:
            ax.table(cellText=model_k.summary.reset_index().values,colLabels=model_k.summary.reset_index().columns,loc='center')
        except:
            pass
        plt.title('Model K: AFT on latent')
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # L
        fig, ax =plt.subplots(figsize=(12,10))
        ax.axis('tight')
        ax.axis('off')
        try:
            ax.table(cellText=model_l.summary.reset_index().values,colLabels=model_l.summary.reset_index().columns,loc='center')
        except:
            pass
        plt.title(f'Model L: AFT on lasso-selected')
        pdf.savefig(bbox_inches='tight')
        plt.close()

        plt.figure()
        scatter = plt.scatter(latent_tsne['reduced_dim0'], latent_tsne['reduced_dim1'], c=latent_tsne['binary'], s=1)
        plt.colorbar(scatter) 
        plt.title('Latent space representation')
        pdf.savefig  (bbox_inches='tight')
        plt.close()  

        ## Make time-dependent AUROC plot for 3 XGB-Cox models:
        fig, ax =plt.subplots(figsize=(8,6))
        try:
            ax.plot(va_times, c_auc, label='XGB_Cox on interim')
            ax.axhline(c_mean_auc, linestyle="--", c=mpl.rcParams['axes.prop_cycle'].by_key()['color'][0])
        except:
            pass
        try:
            ax.plot(va_times, d_auc, label='XGB_Cox on latent')
            ax.axhline(d_mean_auc, linestyle="--", c=mpl.rcParams['axes.prop_cycle'].by_key()['color'][1])
        except:
            pass
        try:
            ax.plot(va_times, e_auc, label='XGB_Cox on lasso-selected')
            ax.axhline(e_mean_auc, linestyle="--", c=mpl.rcParams['axes.prop_cycle'].by_key()['color'][2])
        except:
            pass

        plt.ylim(0,1)
        
        plt.title(f'Comparison of XGBCox for {disease_name}')
        plt.xlabel("days from baseline visit")
        plt.ylabel("time-dependent AUC")
        plt.legend()
        plt.grid(True)
        pdf.savefig(bbox_inches='tight')
        plt.close()

    return ci_a, ci_b, ci_c, ci_d, ci_e, ci_f, ci_g, ci_h, ci_k, ci_l


def predict_evaluate_reduced(disease_name, folder, result_folder, interim_folder, id_file, output_folder, upper_cutoff, letter, store_models=True, seed=None):
    os.makedirs(output_folder, exist_ok=True)
    output_split = output_folder.split('/')[-1]

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    cox_params = {"eta": 0.003, "max_depth": 4, "objective": "survival:cox", "subsample": 0.8}
    aft_params = {'objective': 'survival:aft',
            'eval_metric': 'aft-nloglik',
            'aft_loss_distribution': 'normal',
            'aft_loss_distribution_scale': 2,
            'tree_method': 'hist', 'learning_rate': 0.003, 'max_depth': 4, "subsample": 0.8}
    
    # Prepare latent data
    latent_full, latent_tsne, latent_dim = prepare_latent(disease_name, folder, result_folder, id_file, upper_cutoff, letter)

    # Prepare interim data
    interim_path = f'/mypath/{folder}/{interim_folder}'
    try:
        id_path = f'/mypath/{folder}/data_{letter}/{id_file}.txt'
        interim_full_train, interim_full_test, merged_cols, train_id, test_id = load_interim_new(interim_path, id_path)
    except:
        id_path = f'/mypath/{folder}/data/{id_file}.txt'
        interim_full_train, interim_full_test, merged_cols, train_id, test_id = load_interim_new(interim_path, id_path)
    
    interim_dim = interim_full_train.shape[1]
    interim_full_train, interim_full_test = pd.DataFrame(data=interim_full_train, columns=merged_cols), pd.DataFrame(data=interim_full_test, columns=merged_cols)
    interim_full_train, interim_full_test = pd.concat([train_id.reset_index(drop=True), interim_full_train], ignore_index=False, axis=1), pd.concat([test_id.reset_index(drop=True), interim_full_test], ignore_index=False, axis=1)
    interim_full_train, interim_full_test = prepare_interim(disease_name, folder, result_folder, id_file, upper_cutoff, interim_full_train, interim_full_test)


    # Split latent data into train and test:
    latent_full_train, latent_full_test = latent_full[latent_full['f.eid'].isin(train_id['f.eid'].tolist())], latent_full[latent_full['f.eid'].isin(test_id['f.eid'].tolist())]
    latent_tsne_train, latent_tsne_test = latent_tsne[latent_tsne['f.eid'].isin(train_id['f.eid'].tolist())], latent_tsne[latent_tsne['f.eid'].isin(test_id['f.eid'].tolist())]

    # Set all times to 0 in case model fitting fails
    time_a, time_d, time_g, time_k = 0,0,0,0
    ci_a, ci_d, ci_g, ci_k = 0,0,0,0

    # Model a) CoxPH on latent
    print('----- MODEL A: CoxPH on latent -----')
    model_a = CoxPHFitter()
    ll_cols = [i for i in range(1,latent_dim+1)]+[-2,-1]
    start_time = time.time()
    model_a.fit(latent_full_train.iloc[:,ll_cols], 'delta_days', 'binary',show_progress=True, fit_options={'step_size': 0.1})
    end_time = time.time()
    time_a = end_time - start_time
    #model_a.check_assumptions(latent_full_train.iloc[:,ll_cols])
    #model_a.print_summary()
    ci_a = concordance_index_censored(latent_full_test['binary'].astype(bool), latent_full_test['delta_days'], model_a.predict_partial_hazard(latent_full_test.iloc[:,ll_cols]))[0]
    print('----- MODEL A: C-index on test data: ', ci_a, ' -----')
    if store_models == True:
        pickle_path = f'{output_folder}/model_latent_cph_{output_split}_{disease_name}_{upper_cutoff}.pkl'
        with open(pickle_path, 'wb') as file:
            pickle.dump(model_a, file)


    # Model D: XGBoost with CoxPH on latent space
    print('----- MODEL D: XGBoost with CoxPH on latent space -----')
    y_train, y_test = [r['delta_days'] if r['binary']==1 else -r['delta_days'] for i, r in latent_full_train.iterrows()], [r['delta_days'] if r['binary']==1 else -r['delta_days'] for i, r in latent_full_test.iterrows()]
    xgb_train = xgboost.DMatrix(latent_full_train.iloc[:, [i for i in range(1,latent_dim+1)]].to_numpy(), label=y_train)
    xgb_test = xgboost.DMatrix(latent_full_test.iloc[:, [i for i in range(1,latent_dim+1)]].to_numpy(), label=y_test)
    
    start_time = time.time()
    model_d = xgboost.train(cox_params, xgb_train, 5000, evals=[(xgb_train, "train")], verbose_eval=1000)
    end_time = time.time()
    time_d = end_time - start_time
    ci_d = concordance_index_censored(latent_full_test['binary'].astype(bool), latent_full_test['delta_days'], model_d.predict(xgb_test))[0]
    print('----- MODEL D: C-index on test data: ', ci_d, ' -----')
    d_risk_scores = model_d.predict(xgb_test) #for auroc plot
    if store_models == True:
        pickle_path = f'{output_folder}/model_latent_xgbcph_{output_split}_{disease_name}_{upper_cutoff}.json'
        model_d.save_model(pickle_path)
    try:
        va_times = np.arange(interim_full_test['delta_days'].min(), upper_cutoff, 7)
        d_auc, d_mean_auc = cumulative_dynamic_auc(sksurv.util.Surv.from_dataframe('binary', 'delta_days', latent_full_train), sksurv.util.Surv.from_dataframe('binary', 'delta_days', latent_full_test), d_risk_scores, va_times)
    except:
        d_auc, d_mean_auc = None, None

    
    # Model G: XGBoost AFT model on latent
    print('----- MODEL G: XGBoost with AFT on latent -----')
    y_upper_train, y_upper_test = np.array([+np.inf if r['binary']==0 else r['delta_days'] for i, r in latent_full_train.iterrows()]), np.array([+np.inf if r['binary']==0 else r['delta_days'] for i, r in latent_full_test.iterrows()])
    y_lower_train, y_lower_test = np.array([r['delta_days'] for i, r in latent_full_train.iterrows()]), np.array([r['delta_days'] for i, r in latent_full_test.iterrows()])
    x_train, x_test = latent_full_train.iloc[:, [i for i in range(1,latent_dim+1)]].to_numpy(), latent_full_test.iloc[:, [i for i in range(1,latent_dim+1)]].to_numpy()
    aft_train = xgboost.DMatrix(x_train)
    aft_test = xgboost.DMatrix(x_test)
    aft_train.set_float_info('label_lower_bound', y_lower_train)
    aft_train.set_float_info('label_upper_bound', y_upper_train)
    aft_test.set_float_info('label_lower_bound', y_lower_test)
    aft_test.set_float_info('label_upper_bound', y_upper_test)
    
    start_time = time.time()
    model_g = xgboost.train(aft_params, aft_train, num_boost_round=5000,
                    evals=[(aft_train, 'train')], verbose_eval=1000)
    end_time = time.time()
    time_g = end_time - start_time
    ci_g = concordance_index_censored(latent_full_test['binary'].astype(bool), latent_full_test['delta_days'], -model_g.predict(aft_test))[0]
    print('----- MODEL G: C-index on test data: ', ci_g, ' -----')
    if store_models == True:
        pickle_path = f'{output_folder}/model_latent_xgbaft_{output_split}_{disease_name}_{upper_cutoff}.json'
        model_g.save_model(pickle_path)

    
    # Model K: AFT on latent
    print('----- MODEL K: LogNormalAFTFitter on latent -----')
    model_k = LogNormalAFTFitter()
    ll_cols = [i for i in range(1,latent_dim+1)]+[-2,-1]
    try: 
        start_time = time.time()
        model_k.fit(latent_full_train.iloc[:,ll_cols], 'delta_days', 'binary',show_progress=True, fit_options={'step_size': 0.1})
        end_time = time.time()
        time_k = end_time - start_time
        #model_k.print_summary()
        ci_k = concordance_index_censored(latent_full_test['binary'].astype(bool), latent_full_test['delta_days'], -model_k.predict_expectation(latent_full_test.iloc[:,ll_cols]))[0]
        print('----- MODEL K: C-index on test data: ', ci_k, ' -----')
        if store_models == True:
            pickle_path = f'{output_folder}/model_latent_aft_{output_split}_{disease_name}_{upper_cutoff}.pkl'
            with open(pickle_path, 'wb') as file:
                pickle.dump(model_k, file)
    except: 
        #ci_k = 0
        print('----- MODEL K: FITTING FAILED -----')


    print('---------- C-INDEX SUMMARY ----------')
    print('MODEL A: ', ci_a, time_a)
    print('MODEL D: ', ci_d, time_d)
    print('MODEL G: ', ci_g, time_g)
    print('MODEL K: ', ci_k, time_k)
    
    with PdfPages(f'{output_folder}/report_reduced_{output_split}_{disease_name}_{upper_cutoff}.pdf') as pdf:

        firstPage = plt.figure(figsize=(8,6))
        firstPage.clf()
        title = f'Survival model results for {disease_name} within {upper_cutoff} days from baseline visit:'
        info = f'From: {folder}, {result_folder}'
        txt = f'\
            Model A: CoxPH on latent: C-index A: {round(ci_a,3)} ; fitting took {time_a} \n\
            Model D: XGB_Cox on latent: C-index D: {round(ci_d,3)} ; fitting took {time_d} \n\
            Model G: XGB_AFT on latent: C-index G: {round(ci_g,3)} ; fitting took {time_g} \n\
            Model K: AFT on latent: C-index K: {round(ci_k,3)} ; fitting took {time_k} \n\
                '
        firstPage.text(0.05,0.9, title, transform=firstPage.transFigure, size=12, ha='left')
        firstPage.text(0.05,0.8, info, transform=firstPage.transFigure, size=12, ha='left')
        firstPage.text(0.1,0.3,txt, transform=firstPage.transFigure, size=12, ha='left')
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # A
        fig, ax =plt.subplots(figsize=(12,30))
        ax.axis('tight')
        ax.axis('off')
        ax.table(cellText=model_a.summary.reset_index().values,colLabels=model_a.summary.reset_index().columns,loc='center')
        plt.title('Model A: CoxPH on latent')
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # D
        shap_values = shap.TreeExplainer(model_d).shap_values(latent_full_train.iloc[:, [i for i in range(1,latent_dim+1)]].to_numpy())
        shap.summary_plot(shap_values, latent_full_train.iloc[:, [i for i in range(1,latent_dim+1)]], show=False)
        plt.title('Model D: XGB_Cox on latent')
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # G
        try:
            shap_values = shap.TreeExplainer(model_g).shap_values(latent_full_train.iloc[:, [i for i in range(1,latent_dim+1)]].to_numpy())
            shap.summary_plot(shap_values, latent_full_train.iloc[:, [i for i in range(1,latent_dim+1)]], show=False)
            plt.title('Model G: XGB_AFT on latent')
            pdf.savefig(bbox_inches='tight')
            plt.close()
        except:
            pass

        # K
        fig, ax =plt.subplots(figsize=(12,30))
        ax.axis('tight')
        ax.axis('off')
        try:
            ax.table(cellText=model_k.summary.reset_index().values,colLabels=model_k.summary.reset_index().columns,loc='center')
        except:
            pass
        plt.title('Model K: AFT on latent')
        pdf.savefig(bbox_inches='tight')
        plt.close()

        
        plt.figure()
        scatter = plt.scatter(latent_tsne['reduced_dim0'], latent_tsne['reduced_dim1'], c=latent_tsne['binary'], s=1)
        plt.colorbar(scatter) 
        plt.title('Latent space representation')
        pdf.savefig  (bbox_inches='tight')
        plt.close()  

        ## Make time-dependent AUROC plot for 3 XGB-Cox models:
        fig, ax =plt.subplots(figsize=(8,6))
        
        try:
            ax.plot(va_times, d_auc, label='XGB_Cox on latent')
            ax.axhline(d_mean_auc, linestyle="--", c=mpl.rcParams['axes.prop_cycle'].by_key()['color'][1])
        except:
            pass

        plt.ylim(0,1)
        plt.title(f'XGBCox on latent for {disease_name}')
        plt.xlabel("days from baseline visit")
        plt.ylabel("time-dependent AUC")
        plt.legend()
        plt.grid(True)
        pdf.savefig(bbox_inches='tight')
        plt.close()

    return ci_a, ci_d, ci_g, ci_k


def load_interim_new(interim_path, id_path, split=True):
    print('in load_interim', flush=True)
    file_list = os.listdir(interim_path)
    file_list.remove('indices.pt')
    try:
        file_list.remove('.snakemake_timestamp')
    except:
        pass
    
    indices = torch.load(f'{interim_path}/indices.pt')
    if split is True:
        train, test = indices['train_indices'], indices['test_indices']
    else:
        train, test = indices['train_indices'] + indices['test_indices'], None

    eid = pd.read_csv(id_path, sep=' ', header=None, names=['f.eid'])
    train_id, test_id = eid.iloc[train], eid.iloc[test]

    merged_train, merged_test, merged_cols = None, None, None

    for file in file_list:
        print(f'trying to read file... {file}', flush=True )
        data = torch.load(f'{interim_path}/{file}')
        interim = data['tensor']
        if len(interim.shape)>2 :
            interim = interim.argmax(dim=-1)
        interim_train, interim_test = interim[train,:], interim[test,:]
        
        if merged_train is None and len(interim.shape)<3:
            merged_train, merged_test, merged_cols = interim_train, interim_test, data['feature_names']
        else:
            merged_train, merged_test, merged_cols = torch.cat((merged_train, interim_train), dim=1), torch.cat((merged_test, interim_test), dim=1), merged_cols+data['feature_names']
        
    return merged_train, merged_test, merged_cols, train_id, test_id


def prepare_latent(disease_name, folder, result_folder, id_file, upper_cutoff, letter):
    vd = pd.read_csv('/mypath/diseases/visit_date.tsv', sep='\t')
    dd = pd.read_csv('/mypath/diseases/death_date.tsv', sep='\t')
    lf = pd.read_csv('/mypath/patients/lost_followup.tsv', sep='\t')
    disease = pd.read_csv(f'/mypath/diseases/icd10_filtered/filtered_{disease_name}.tsv', sep='\t')
    self_reported = pd.read_csv(f'/mypath/diseases/icd10_filtered/self_reported_{disease_name}.tsv', sep='\t')
    self_reported = list(self_reported['f.eid'])
    blacklist = pd.read_csv('/mypath/diseases/langenberg_exclusion/all_blacklist.tsv', sep='\t')
    blacklist = list(blacklist['f.eid'])
    latent = pd.read_csv(f'/mypath/{folder}/{result_folder}/latent_space/latent_space.csv')
    try:
        ids = pd.read_csv(f'/mypath/{folder}/data_{letter}/{id_file}.txt', sep=' ', header=None, names=['f.eid'])
    except:
        ids = pd.read_csv(f'/mypath/{folder}/data/{id_file}.txt', sep=' ', header=None, names=['f.eid'])

    latent_dim = latent.shape[1]-2
    tsne = latent[latent.columns[:2]]
    tsne = ids.join(tsne)
    latent = latent[latent.columns[2:]]
    latent = ids.join(latent)

    vd_dd = pd.merge(vd, dd, on='f.eid', how='inner')
    vd_dd_lf = pd.merge(vd_dd, lf, on='f.eid', how='inner')
    filter_time = pd.merge(vd_dd_lf, disease, how='left', on='f.eid')
    filter_time['date'] = pd.to_datetime(filter_time['date'])
    filter_time['f.53.0.0'] = pd.to_datetime(filter_time['f.53.0.0'])
    filter_time['f.191.0.0'] = pd.to_datetime(filter_time['f.191.0.0'])
    filter_time['f.40000.0.0'] = pd.to_datetime(filter_time['f.40000.0.0'])
    # delta: time between UKB visit and diagnosis
    filter_time['delta'] = filter_time['date'].sub(filter_time['f.53.0.0'], axis=0)
    # death_delta: time between UKB visit and death
    filter_time['death_delta'] = filter_time['f.40000.0.0'].sub(filter_time['f.53.0.0'], axis=0).dt.days
    filter_time['lf_delta'] = filter_time['f.191.0.0'].sub(filter_time['f.53.0.0'], axis=0).dt.days
    filter_time['delta_days'] = filter_time['delta'].dt.days
    # For all undiagnosed patients, set delta days to upper cutoff + 1
    filter_time['delta_days'] = filter_time['delta_days'].fillna(upper_cutoff+1)
    # List of patients which were diagnosed too early or died too early
    del_patients = filter_time[filter_time['delta_days']<=180]['f.eid'].tolist()
    del_patients = del_patients + filter_time[filter_time['death_delta']<=180]['f.eid'].tolist()
    # All patients who died after UKB visit and before upper cutoff: 
    death_patients = filter_time[(filter_time['f.40000.0.0'].sub(filter_time['f.53.0.0'], axis=0).dt.days < upper_cutoff) & (filter_time['f.40000.0.0'].sub(filter_time['f.53.0.0'], axis=0).dt.days > 0)]['f.eid'].tolist()
    
    full_merge = pd.merge(latent, filter_time, how='inner', on='f.eid')
    # Only keep patients that were diagnosed after more than 180 days or not diagnosed at all
    full_merge = full_merge[full_merge['delta_days']>180]
    # Binary indicator whether patient was diagnosed before upper cutoff
    full_merge['binary'] = full_merge['delta_days']<=upper_cutoff
    full_merge['binary'] = full_merge['binary'].astype(int)
    # Set undiagnosed patient's delta days to upper cutoff (prev. was upper_cutoff + 1)
    full_merge.loc[full_merge['binary'] == 0, 'delta_days'] = upper_cutoff
    # Set death delta as censor date for patients that died without diagnosis
    full_merge.loc[(full_merge['death_delta'] > 0) & (full_merge['death_delta'] < upper_cutoff) & (pd.isna(full_merge['date'])), 'delta_days'] = full_merge['death_delta']
    full_merge.loc[(full_merge['lf_delta'] > 0) & (full_merge['lf_delta'] < upper_cutoff) & (pd.isna(full_merge['date'])), 'delta_days'] = full_merge['lf_delta']

    # Remove patients that meet certain conditions: del_patients from above, blacklist from langenberg, self_reports: self-reported disease at baseline visit.
    full_merge = full_merge[~full_merge['f.eid'].isin(del_patients)]
    full_merge = full_merge[~full_merge['f.eid'].isin(blacklist)]
    full_merge = full_merge[~full_merge['f.eid'].isin(self_reported)]

    tsne_cols = tsne.columns.tolist() +['delta_days', 'binary']
    tsne_merge = pd.merge(tsne, full_merge, how='inner', on='f.eid')
    tsne_merge = tsne_merge[tsne_cols]
    return full_merge, tsne_merge, latent_dim

def prepare_interim(disease_name, folder, result_folder, id_file, upper_cutoff, interim_train, interim_test):
    vd = pd.read_csv('/mypath/diseases/visit_date.tsv', sep='\t')
    dd = pd.read_csv('/mypath/diseases/death_date.tsv', sep='\t')
    lf = pd.read_csv('/mypath/patients/lost_followup.tsv', sep='\t')
    disease = pd.read_csv(f'/mypath/diseases/icd10_filtered/filtered_{disease_name}.tsv', sep='\t')
    self_reported = pd.read_csv(f'/mypath/diseases/icd10_filtered/self_reported_{disease_name}.tsv', sep='\t')
    blacklist = pd.read_csv('/mypath/diseases/langenberg_exclusion/all_blacklist.tsv', sep='\t')
    blacklist = list(blacklist['f.eid'])
    self_reported = list(self_reported['f.eid'])

    vd_dd = pd.merge(vd, dd, on='f.eid', how='inner')
    vd_dd_lf = pd.merge(vd_dd, lf, on='f.eid', how='inner')
    filter_time = pd.merge(vd_dd_lf, disease, how='left', on='f.eid')
    filter_time['date'] = pd.to_datetime(filter_time['date'])
    filter_time['f.53.0.0'] = pd.to_datetime(filter_time['f.53.0.0'])
    filter_time['f.191.0.0'] = pd.to_datetime(filter_time['f.191.0.0'])
    filter_time['f.40000.0.0'] = pd.to_datetime(filter_time['f.40000.0.0'])
    # delta: time between UKB visit and diagnosis
    filter_time['delta'] = filter_time['date'].sub(filter_time['f.53.0.0'], axis=0)
    # death_delta: time between UKB visit and death
    filter_time['death_delta'] = filter_time['f.40000.0.0'].sub(filter_time['f.53.0.0'], axis=0).dt.days
    filter_time['lf_delta'] = filter_time['f.191.0.0'].sub(filter_time['f.53.0.0'], axis=0).dt.days
    filter_time['delta_days'] = filter_time['delta'].dt.days
    # For all undiagnosed patients, set delta days to upper cutoff + 1
    filter_time['delta_days'] = filter_time['delta_days'].fillna(upper_cutoff+1)
    # List of patients which were diagnosed too early or died too early
    del_patients = filter_time[filter_time['delta_days']<=180]['f.eid'].tolist()
    del_patients = del_patients + filter_time[filter_time['death_delta']<=180]['f.eid'].tolist()
    # All patients who died after UKB visit and before upper cutoff: 
    death_patients = filter_time[(filter_time['f.40000.0.0'].sub(filter_time['f.53.0.0'], axis=0).dt.days < upper_cutoff) & (filter_time['f.40000.0.0'].sub(filter_time['f.53.0.0'], axis=0).dt.days > 0)]['f.eid'].tolist()
    
    # TRAIN
    full_merge_train = pd.merge(interim_train, filter_time, how='inner', on='f.eid')
    # Only keep patients that were diagnosed after more than 180 days or not diagnosed at all
    full_merge_train = full_merge_train[full_merge_train['delta_days']>180]
    # Binary indicator whether patient was diagnosed before upper cutoff
    full_merge_train['binary'] = full_merge_train['delta_days']<=upper_cutoff
    full_merge_train['binary'] = full_merge_train['binary'].astype(int)
    # Set undiagnosed patient's delta days to upper cutoff (prev. was upper_cutoff + 1)
    full_merge_train.loc[full_merge_train['binary'] == 0, 'delta_days'] = upper_cutoff
    # Set death delta as censor date for patients that died without diagnosis
    full_merge_train.loc[(full_merge_train['death_delta'] > 0) & (full_merge_train['death_delta'] < upper_cutoff) & (pd.isna(full_merge_train['date'])), 'delta_days'] = full_merge_train['death_delta']
    full_merge_train.loc[(full_merge_train['lf_delta'] > 0) & (full_merge_train['lf_delta'] < upper_cutoff) & (pd.isna(full_merge_train['date'])), 'delta_days'] = full_merge_train['lf_delta']

    # Remove patients that meet certain conditions: del_patients from above, blacklist from langenberg, self_reports: self-reported disease at baseline visit.
    full_merge_train = full_merge_train[~full_merge_train['f.eid'].isin(del_patients)]
    full_merge_train = full_merge_train[~full_merge_train['f.eid'].isin(blacklist)]
    full_merge_train = full_merge_train[~full_merge_train['f.eid'].isin(self_reported)]

    # TEST
    full_merge_test = pd.merge(interim_test, filter_time, how='inner', on='f.eid')
    # Only keep patients that were diagnosed after more than 180 days or not diagnosed at all
    full_merge_test = full_merge_test[full_merge_test['delta_days']>180]
    # Binary indicator whether patient was diagnosed before upper cutoff
    full_merge_test['binary'] = full_merge_test['delta_days']<=upper_cutoff
    full_merge_test['binary'] = full_merge_test['binary'].astype(int)
    # Set undiagnosed patient's delta days to upper cutoff (prev. was upper_cutoff + 1)
    full_merge_test.loc[full_merge_test['binary'] == 0, 'delta_days'] = upper_cutoff
    # Set death delta as censor date for patients that died without diagnosis
    full_merge_test.loc[(full_merge_test['death_delta'] > 0) & (full_merge_test['death_delta'] < upper_cutoff) & (pd.isna(full_merge_test['date'])), 'delta_days'] = full_merge_test['death_delta']
    full_merge_test.loc[(full_merge_test['lf_delta'] > 0) & (full_merge_test['lf_delta'] < upper_cutoff) & (pd.isna(full_merge_test['date'])), 'delta_days'] = full_merge_test['lf_delta']

    # Remove patients that meet certain conditions: del_patients from above, blacklist from langenberg, self_reports: self-reported disease at baseline visit.
    full_merge_test = full_merge_test[~full_merge_test['f.eid'].isin(del_patients)]
    full_merge_test = full_merge_test[~full_merge_test['f.eid'].isin(blacklist)]
    full_merge_test = full_merge_test[~full_merge_test['f.eid'].isin(self_reported)]

    return full_merge_train, full_merge_test


if __name__ == "__main__":
    disease_name = sys.argv[1]
    folder = sys.argv[2]
    result_folder = sys.argv[3]
    interim_folder = sys.argv[4]
    id_file = sys.argv[5]
    output_folder = sys.argv[6]
    upper_cutoff = int(sys.argv[7])
    store_models = bool(sys.argv[8])
    try:
        seed = int(sys.argv[9])
    except:
        seed = None
    letter = sys.argv[10]

    if disease_name == 'all':
        run_all(folder, result_folder, interim_folder, id_file, output_folder, upper_cutoff, letter, store_models, seed)
    elif disease_name == 'reduced':
        run_reduced(folder, result_folder, interim_folder, id_file, output_folder, upper_cutoff, letter, store_models, seed)
    else:
        predict_evaluate(disease_name, folder, result_folder, interim_folder, id_file, output_folder, upper_cutoff, letter, store_models, seed)