import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import torch


folder = 'prep3'
size = 100000

latent_with = pd.read_csv(f'/mypath/{folder}/results_nores_100000_H/latent_space/latent_space.csv')
latent_res = pd.read_csv(f'/mypath/{folder}/results_res_100000_H/latent_space/latent_space.csv')
latent_0cov = pd.read_csv(f'/mypath/{folder}/results_0cov_100000_H/latent_space/latent_space.csv')

tsne_with = latent_with.iloc[:,:2]
tsne_res = latent_res.iloc[:,:2]
tsne_0cov = latent_0cov.iloc[:,:2]

latent_with = latent_with.iloc[:,2:]
latent_res = latent_res.iloc[:,2:]
latent_0cov = latent_0cov.iloc[:,2:]

k_list = [10,100, 1000, 10000, 99999]
score_df = pd.DataFrame(columns=['name']+k_list)
for covariate in ['assessment_centre', 'ethnic_background', 'sex', 'smoking_status', 'alcohol_frequency']:
    scores_with, scores_res, scores_0cov = [], [], []
    for k in k_list:    
        
        cov = torch.load(f'/mypath/{folder}/interim_data_nores_{size}_H/{covariate}_{size}.pt', weights_only=False)
        print(f'{covariate} with k={k}')
        indices = torch.argmax(cov['tensor'].squeeze(1), dim=1)
        inv_map = {v: k for k, v in cov['mapping'].items()}
        labels = [int(inv_map[i]) for i in indices.tolist()]

        neigh_with = KNeighborsClassifier(n_neighbors=k)
        neigh_with.fit(latent_with, labels)
        score = neigh_with.score(latent_with, labels)
        print('Not residualized: ', score)
        scores_with = scores_with+[score]

        neigh_res = KNeighborsClassifier(n_neighbors=k)
        neigh_res.fit(latent_res, labels)
        score = neigh_res.score(latent_res, labels)
        print('Residualized: ', score)
        scores_res = scores_res+[score]

        neigh_0cov = KNeighborsClassifier(n_neighbors=k)
        neigh_0cov.fit(latent_0cov, labels)
        score = neigh_0cov.score(latent_0cov, labels)
        print('0cov: ', score)
        scores_0cov = scores_0cov+[score]
        

    score_df.loc[len(score_df.index)] = [f'{covariate}_nores']+scores_with
    score_df.loc[len(score_df.index)] = [f'{covariate}_withres']+scores_res
    score_df.loc[len(score_df.index)] = [f'{covariate}_0cov']+scores_0cov

score_df.to_csv(f'/mypath/baseline/KNN_comparison/knn_scores_{folder}_{size}')