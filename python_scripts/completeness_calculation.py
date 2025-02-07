import numpy as np
import pandas as pd
import random
import sys
from preprocessing_functions import get_id_from_tsv, sample_store
from sklearn.model_selection import train_test_split


def get_snp_patients(folder_name, reduce_proteomics, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    compl = pd.read_csv(f'/mypath/patients/unrelated_notna.tsv', sep='\t')
    all_ids = compl['f.eid'].to_list()

    # Load all genotype files, keep only sampled ids, split them.
    all_genomics = pd.DataFrame({'f.eid': all_ids})
    for chrn in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]:
        
        chrdf = pd.read_csv(f'/mypath/genomics/step2/data/extracted_genotypes/chr{chrn}/chr{chrn}_prep.tsv', sep='\t')
        all_genomics = pd.merge(all_genomics, chrdf, on='f.eid', how='left')
    gen_ids = all_genomics.loc[all_genomics.iloc[:, 1:].notna().any(axis=1), 'f.eid'].tolist()


    all_pqtl = pd.DataFrame({'f.eid': all_ids})
    for chrn in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]:
        chrdf = pd.read_csv(f'/mypath/genomics/step2/data/pQTL_extracted_genotypes/chr{chrn}/chr{chrn}_prep.tsv', sep='\t')
        all_pqtl = pd.merge(all_pqtl, chrdf, on='f.eid', how='left')
    pqtl_ids = all_pqtl.loc[all_pqtl.iloc[:, 1:].notna().any(axis=1), 'f.eid'].tolist()
    intersection = list(set(gen_ids) & set(pqtl_ids))

    i_df = pd.DataFrame(intersection, columns=['f.eid'])
    i_df.to_csv('/mypath/patients/with_snp.tsv', sep='\t', index=False)




def completeness_per_patient(folder_name, reduce_proteomics, seed=None):
    '''
    Set paths to raw data: 
    - phenotypes_path contains anthropometrics, physical measurments, metabolomics, blood measurments, lifestyle factors, ...
    - olink_path contains proteomics data
    '''
    phenotypes_path = '/path_to_data/ukb677323.tab.gz'
    olink_path = '/path_to_data/olink_data.txt.gz'


    '''
    Set chunksize to process big data files and set random seed for patient sampling. 
    '''
    chunks = 10000
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    '''
    For every category from the phenotype data, get the list of feature IDs, prepend the eID and fomat the names such that they match the raw data columns.
    Finally, merge all the feature names such that the raw data only has to be read once.
    '''
    id_batch = pd.read_csv(f'/mypath/{folder_name}/preprocessing/batch.tsv', sep='\t')
    id_blood_biochem = pd.read_csv(f'/mypath/{folder_name}/preprocessing/blood_biochemistry.tsv', sep='\t')
    id_blood_count = pd.read_csv(f'/mypath/{folder_name}/preprocessing/blood_count.tsv', sep='\t')
    id_infect = pd.read_csv(f'/mypath/{folder_name}/preprocessing/infectious_diseases.tsv', sep='\t')
    id_metabolomics = pd.read_csv(f'/mypath/{folder_name}/preprocessing/metabolomics.tsv', sep='\t')
    id_physical = pd.read_csv(f'/mypath/{folder_name}/preprocessing/physical_measurements.tsv', sep='\t')
    id_pop_char = pd.read_csv(f'/mypath/{folder_name}/preprocessing/population_characteristics.tsv', sep='\t')
    id_urine = pd.read_csv(f'/mypath/{folder_name}/preprocessing/urine.tsv', sep='\t')
    
    mapping = pd.concat([id_batch, id_blood_biochem, id_blood_count, id_infect, id_metabolomics, id_physical, id_pop_char, id_urine])
    mapping['id'] = ['f.' +str(mapping['id'].values[i])+ '.0.0' for i in range(len(mapping['id']))]
    rename_dict = pd.Series(mapping['name'].values, index=mapping['id']).to_dict()
    print(mapping, rename_dict)

    age = [str(id_pop_char[id_pop_char['name']=='Age when attended assessment centre']['id'].values[0])]
    age = ['f.eid']+['f.' + col + '.0.0' for col in age]

    fasting = [str(id_pop_char[id_pop_char['name']=='Fasting time']['id'].values[0])]
    fasting = ['f.eid']+['f.' + col + '.0.0' for col in fasting]
    
    sex = [str(id_pop_char[id_pop_char['name']=='Sex']['id'].values[0])]
    sex = ['f.eid']+['f.' + col + '.0.0' for col in sex]
    
    assessment_centre = [str(id_pop_char[id_pop_char['name']=='UKB assessment centre']['id'].values[0])]
    assessment_centre = ['f.eid']+['f.' + col + '.0.0' for col in assessment_centre]

    ethnic_bg = [str(id_pop_char[id_pop_char['name']=='Ethnic background']['id'].values[0])]
    ethnic_bg = ['f.eid']+['f.' + col + '.0.0' for col in ethnic_bg]

    smoking = [str(id_pop_char[id_pop_char['name']=='Smoking status']['id'].values[0])]
    smoking = ['f.eid']+['f.' + col + '.0.0' for col in smoking]

    alcohol = [str(id_pop_char[id_pop_char['name']=='Alcohol intake frequency']['id'].values[0])]
    alcohol = ['f.eid']+['f.' + col + '.0.0' for col in alcohol]

    physical_measurments = [str(id_physical['id'].values[i]) for i in range(len(id_physical['id']))]
    physical_measurments = ['f.eid']+['f.' + col + '.0.0' for col in physical_measurments]

    blood_count = [str(id_blood_count['id'].values[i]) for i in range(len(id_blood_count['id']))]
    blood_count = ['f.eid']+['f.' + col + '.0.0' for col in blood_count]

    blood_biochem = [str(id_blood_biochem['id'].values[i]) for i in range(len(id_blood_biochem['id']))]
    blood_biochem = ['f.eid']+['f.' + col + '.0.0' for col in blood_biochem]

    infect = [str(id_infect['id'].values[i]) for i in range(len(id_infect['id']))]
    infect = ['f.eid']+['f.' + col + '.0.0' for col in infect]

    metabolomics = [str(id_metabolomics['id'].values[i]) for i in range(len(id_metabolomics['id']))]
    metabolomics = ['f.eid']+['f.' + col + '.0.0' for col in metabolomics]

    urine = [str(id_urine['id'].values[i]) for i in range(len(id_urine['id']))]
    urine = ['f.eid']+['f.' + col + '.0.0' for col in urine]

    merged_columns = ['f.eid'] + age[1:] + fasting[1:] + sex[1:] + assessment_centre[1:] + ethnic_bg[1:] + physical_measurments[1:] + smoking[1:] + alcohol[1:] + blood_count[1:] + blood_biochem[1:] + infect[1:] + metabolomics[1:] + urine[1:] 
    reduced_cols = ['f.eid'] + physical_measurments[1:] + blood_count[1:] + blood_biochem[1:] + infect[1:] + metabolomics[1:] + urine[1:] 



    '''
    Make a big dataframe and fill it with all chosen features.
    Afterwards split the merged df into smaller ones which have been defined as above.
    Finally, retrieve a list of all eIDs and randomly sample a subset of them.
    '''
    merged = pd.DataFrame()
    for chunk in pd.read_csv(phenotypes_path, chunksize=chunks, sep='\t', compression='gzip', usecols=merged_columns):
        merged = pd.concat([merged, chunk])

    '''
    Split the merged df accoding to the defined data categories above and rename the columns according to the mapping.
    '''
    age_df = merged[age]
    age_df = age_df.rename(columns=rename_dict, inplace=False)
    fasting_df = merged[fasting]
    fasting_df = fasting_df.rename(columns=rename_dict, inplace=False)
    sex_df = merged[sex]
    sex_df = sex_df.rename(columns=rename_dict, inplace=False)
    ac_df = merged[assessment_centre]
    ac_df = ac_df.rename(columns=rename_dict, inplace=False)
    ebg_df = merged[ethnic_bg]
    ebg_df = ebg_df.rename(columns=rename_dict, inplace=False)
    sm_df = merged[smoking]
    sm_df = sm_df.rename(columns=rename_dict, inplace=False)
    al_df = merged[alcohol]
    al_df = al_df.rename(columns=rename_dict, inplace=False)
    pm_df = merged[physical_measurments]
    pm_df = pm_df.rename(columns=rename_dict, inplace=False)
    bc_df = merged[blood_count]
    bc_df = bc_df.rename(columns=rename_dict, inplace=False)
    bb_df = merged[blood_biochem]
    bb_df = bb_df.rename(columns=rename_dict, inplace=False)
    inf_df = merged[infect]
    inf_df = inf_df.rename(columns=rename_dict, inplace=False)
    urine_df = merged[urine]
    urine_df = urine_df.rename(columns=rename_dict, inplace=False)
    met_df = merged[metabolomics]
    met_df = met_df.rename(columns=rename_dict, inplace=False)

    
    prot = pd.read_csv(olink_path, sep='\t', compression='gzip')
    prot = prot[prot['ins_index'] == 0]
    prot = prot.pivot(index='eid', columns='protein_id', values='result')
    prot.reset_index(inplace=True)
    prot.columns.name = None

    # merge proteomics with sex to get all ids into proteomics and then only retain proteomic columns
    prot_allids = pd.merge(prot, sex_df, left_on='eid', right_on='f.eid', how='right')
    #prot = prot_allids[prot.columns]
    prot = prot_allids.drop(['eid', 'Sex'], axis=1)
    print('prot shape', prot.shape)
    cols = prot.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    prot = prot[cols]

    prot_encoding = pd.read_csv('/mypath/olink/encoding143.tsv', sep='\t')
    prot_encoding['abr'] = prot_encoding['meaning'].str.split(';').str[0]
    encoding_dict = dict(zip(prot_encoding['coding'], prot_encoding['abr']))
    prot = prot.rename(columns=encoding_dict, inplace=False)

    if reduce_proteomics == True:
        top100 = pd.read_csv('/mypath/olink/over0_5_comparison.tsv', sep='\t', header=None, names=['abr'])
        top100 = ['f.eid'] + top100['abr'].to_list()
        valid_columns = prot.columns.intersection(top100)
        prot = prot[valid_columns]
    
    
    # Load list of patients that are not related, then sample from those
    not_related = pd.read_csv('/mypath/patients/all_pheno_minus_out.txt', sep=' ')
    withdrawal = pd.read_csv('/mypath/patients/withdrawal.txt', sep=' ')
    eids = not_related[~not_related['f.eid'].isin(withdrawal['f.eid'])]['f.eid'].tolist()
    print('TOTAL NUMBER OF IDS TO SAMPLE FROM: ', len(eids))
    print('not related: ', len(not_related), 'withdrawal: ', len(withdrawal))
    sampled_ids = eids

    # Load all genotype files, keep only sampled ids, split them.
    all_genomics = pd.DataFrame({'f.eid': sampled_ids})
    for chrn in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]:
        chrdf = pd.read_csv(f'/mypath/genomics/step2/data/extracted_genotypes/chr{chrn}/chr{chrn}_prep.tsv', sep='\t')
        all_genomics = pd.merge(all_genomics, chrdf, on='f.eid', how='left')


    all_pqtl = pd.DataFrame({'f.eid': sampled_ids})
    for chrn in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]:
        chrdf = pd.read_csv(f'/mypath/genomics/step2/data/pQTL_extracted_genotypes/chr{chrn}/chr{chrn}_prep.tsv', sep='\t')
        all_pqtl = pd.merge(all_pqtl, chrdf, on='f.eid', how='left')


    ### Calculate missingness per patient
    merged = merged.set_index('f.eid').reindex(sorted(eids)).reset_index()
    prot = prot.set_index('f.eid').reindex(sorted(eids)).reset_index()
    all_genomics = all_genomics.set_index('f.eid').reindex(sorted(eids)).reset_index()
    all_pqtl = all_pqtl.set_index('f.eid').reindex(sorted(eids)).reset_index()


    notna_patients = merged[[merged.columns[0]]]
    notna_patients['notna'] = (prot[prot.columns[1:]].notna().sum(axis=1) + merged[reduced_cols[1:]].notna().sum(axis=1) + all_genomics[all_genomics.columns[1:]].notna().sum(axis=1) + all_pqtl[all_pqtl.columns[1:]].notna().sum(axis=1))  / (prot[prot.columns[1:]].notna().count(axis=1) + merged[reduced_cols[1:]].notna().count(axis=1) + all_genomics[all_genomics.columns[1:]].notna().count(axis=1) + all_pqtl[all_pqtl.columns[1:]].notna().count(axis=1))
    notna_patients.to_csv('/mypath/patients/unrelated_notna.tsv', sep='\t', index=False)

    notna_excl_gen = merged[[merged.columns[0]]]
    notna_excl_gen['notna'] = (prot[prot.columns[1:]].notna().sum(axis=1) + merged[reduced_cols[1:]].notna().sum(axis=1))  / (prot[prot.columns[1:]].notna().count(axis=1) + merged[reduced_cols[1:]].notna().count(axis=1))
    notna_excl_gen.to_csv('/mypath/patients/unrelated_notna_excl_snp.tsv', sep='\t', index=False)

    notna_excl_prot = merged[[merged.columns[0]]]
    notna_excl_prot['notna'] = (merged[reduced_cols].notna().sum(axis=1) + all_genomics[all_genomics.columns[1:]].notna().sum(axis=1) + all_pqtl[all_pqtl.columns[1:]].notna().sum(axis=1))  / (merged[reduced_cols[1:]].notna().count(axis=1) + all_genomics[all_genomics.columns[1:]].notna().count(axis=1) + all_pqtl[all_pqtl.columns[1:]].notna().count(axis=1))
    notna_excl_prot.to_csv('/mypath/patients/unrelated_notna_excl_prot.tsv', sep='\t', index=False)


def completeness_per_feature(folder_name, reduce_proteomics, seed=None):
    '''
    Set paths to raw data: 
    - phenotypes_path contains anthropometrics, physical measurments, metabolomics, blood measurments, lifestyle factors, ...
    - olink_path contains proteomics data
    '''
    phenotypes_path = '/path_to_data/ukb677323.tab.gz'
    olink_path = '/path_to_data/olink_data.txt.gz'


    '''
    Set chunksize to process big data files and set random seed for patient sampling. 
    '''
    chunks = 10000
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    '''
    For every category from the phenotype data, get the list of feature IDs, prepend the eID and fomat the names such that they match the raw data columns.
    Finally, merge all the feature names such that the raw data only has to be read once.
    '''
    id_batch = pd.read_csv(f'/mypath/{folder_name}/preprocessing/batch.tsv', sep='\t')
    id_blood_biochem = pd.read_csv(f'/mypath/{folder_name}/preprocessing/blood_biochemistry.tsv', sep='\t')
    id_blood_count = pd.read_csv(f'/mypath/{folder_name}/preprocessing/blood_count.tsv', sep='\t')
    id_infect = pd.read_csv(f'/mypath/{folder_name}/preprocessing/infectious_diseases.tsv', sep='\t')
    id_metabolomics = pd.read_csv(f'/mypath/{folder_name}/preprocessing/metabolomics.tsv', sep='\t')
    id_physical = pd.read_csv(f'/mypath/{folder_name}/preprocessing/physical_measurements.tsv', sep='\t')
    id_pop_char = pd.read_csv(f'/mypath/{folder_name}/preprocessing/population_characteristics.tsv', sep='\t')
    id_urine = pd.read_csv(f'/mypath/{folder_name}/preprocessing/urine.tsv', sep='\t')
    
    mapping = pd.concat([id_batch, id_blood_biochem, id_blood_count, id_infect, id_metabolomics, id_physical, id_pop_char, id_urine])
    mapping['id'] = ['f.' +str(mapping['id'].values[i])+ '.0.0' for i in range(len(mapping['id']))]
    rename_dict = pd.Series(mapping['name'].values, index=mapping['id']).to_dict()
    print(mapping, rename_dict)

    age = [str(id_pop_char[id_pop_char['name']=='Age when attended assessment centre']['id'].values[0])]
    age = ['f.eid']+['f.' + col + '.0.0' for col in age]

    fasting = [str(id_pop_char[id_pop_char['name']=='Fasting time']['id'].values[0])]
    fasting = ['f.eid']+['f.' + col + '.0.0' for col in fasting]
    
    sex = [str(id_pop_char[id_pop_char['name']=='Sex']['id'].values[0])]
    sex = ['f.eid']+['f.' + col + '.0.0' for col in sex]
    
    assessment_centre = [str(id_pop_char[id_pop_char['name']=='UKB assessment centre']['id'].values[0])]
    assessment_centre = ['f.eid']+['f.' + col + '.0.0' for col in assessment_centre]

    ethnic_bg = [str(id_pop_char[id_pop_char['name']=='Ethnic background']['id'].values[0])]
    ethnic_bg = ['f.eid']+['f.' + col + '.0.0' for col in ethnic_bg]

    smoking = [str(id_pop_char[id_pop_char['name']=='Smoking status']['id'].values[0])]
    smoking = ['f.eid']+['f.' + col + '.0.0' for col in smoking]

    alcohol = [str(id_pop_char[id_pop_char['name']=='Alcohol intake frequency']['id'].values[0])]
    alcohol = ['f.eid']+['f.' + col + '.0.0' for col in alcohol]

    physical_measurments = [str(id_physical['id'].values[i]) for i in range(len(id_physical['id']))]
    physical_measurments = ['f.eid']+['f.' + col + '.0.0' for col in physical_measurments]

    blood_count = [str(id_blood_count['id'].values[i]) for i in range(len(id_blood_count['id']))]
    blood_count = ['f.eid']+['f.' + col + '.0.0' for col in blood_count]

    blood_biochem = [str(id_blood_biochem['id'].values[i]) for i in range(len(id_blood_biochem['id']))]
    blood_biochem = ['f.eid']+['f.' + col + '.0.0' for col in blood_biochem]

    infect = [str(id_infect['id'].values[i]) for i in range(len(id_infect['id']))]
    infect = ['f.eid']+['f.' + col + '.0.0' for col in infect]

    metabolomics = [str(id_metabolomics['id'].values[i]) for i in range(len(id_metabolomics['id']))]
    metabolomics = ['f.eid']+['f.' + col + '.0.0' for col in metabolomics]

    urine = [str(id_urine['id'].values[i]) for i in range(len(id_urine['id']))]
    urine = ['f.eid']+['f.' + col + '.0.0' for col in urine]

    merged_columns = ['f.eid'] + age[1:] + fasting[1:] + sex[1:] + assessment_centre[1:] + ethnic_bg[1:] + physical_measurments[1:] + smoking[1:] + alcohol[1:] + blood_count[1:] + blood_biochem[1:] + infect[1:] + metabolomics[1:] + urine[1:] 
    reduced_cols = ['f.eid'] + blood_count[1:] + blood_biochem[1:] + infect[1:] + metabolomics[1:] + urine[1:] 



    '''
    Make a big dataframe and fill it with all chosen features.
    Afterwards split the merged df into smaller ones which have been defined as above.
    Finally, retrieve a list of all eIDs and randomly sample a subset of them.
    '''
    merged = pd.DataFrame()
    for chunk in pd.read_csv(phenotypes_path, chunksize=chunks, sep='\t', compression='gzip', usecols=merged_columns):
        merged = pd.concat([merged, chunk])

    '''
    Split the merged df accoding to the defined data categories above and rename the columns according to the mapping.
    '''
    age_df = merged[age]
    age_df = age_df.rename(columns=rename_dict, inplace=False)
    fasting_df = merged[fasting]
    fasting_df = fasting_df.rename(columns=rename_dict, inplace=False)
    sex_df = merged[sex]
    sex_df = sex_df.rename(columns=rename_dict, inplace=False)
    ac_df = merged[assessment_centre]
    ac_df = ac_df.rename(columns=rename_dict, inplace=False)
    ebg_df = merged[ethnic_bg]
    ebg_df = ebg_df.rename(columns=rename_dict, inplace=False)
    sm_df = merged[smoking]
    sm_df = sm_df.rename(columns=rename_dict, inplace=False)
    al_df = merged[alcohol]
    al_df = al_df.rename(columns=rename_dict, inplace=False)
    pm_df = merged[physical_measurments]
    pm_df = pm_df.rename(columns=rename_dict, inplace=False)
    bc_df = merged[blood_count]
    bc_df = bc_df.rename(columns=rename_dict, inplace=False)
    bb_df = merged[blood_biochem]
    bb_df = bb_df.rename(columns=rename_dict, inplace=False)
    inf_df = merged[infect]
    inf_df = inf_df.rename(columns=rename_dict, inplace=False)
    urine_df = merged[urine]
    urine_df = urine_df.rename(columns=rename_dict, inplace=False)
    met_df = merged[metabolomics]
    met_df = met_df.rename(columns=rename_dict, inplace=False)

    
    prot = pd.read_csv(olink_path, sep='\t', compression='gzip')
    prot = prot[prot['ins_index'] == 0]
    prot = prot.pivot(index='eid', columns='protein_id', values='result')
    prot.reset_index(inplace=True)
    prot.columns.name = None

    # merge proteomics with sex to get all ids into proteomics and then only retain proteomic columns
    prot_allids = pd.merge(prot, sex_df, left_on='eid', right_on='f.eid', how='right')
    #prot = prot_allids[prot.columns]
    prot = prot_allids.drop(['eid', 'Sex'], axis=1)
    print('prot shape', prot.shape)
    cols = prot.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    prot = prot[cols]

    prot_encoding = pd.read_csv('/mypath/olink/encoding143.tsv', sep='\t')
    prot_encoding['abr'] = prot_encoding['meaning'].str.split(';').str[0]
    encoding_dict = dict(zip(prot_encoding['coding'], prot_encoding['abr']))
    prot = prot.rename(columns=encoding_dict, inplace=False)

    if reduce_proteomics == True:
        top100 = pd.read_csv('/mypath/olink/over0_5_comparison.tsv', sep='\t', header=None, names=['abr'])
        top100 = ['f.eid'] + top100['abr'].to_list()
        valid_columns = prot.columns.intersection(top100)
        prot = prot[valid_columns]
    
    
    # Load list of patients that are not related, then sample from those
    not_related = pd.read_csv('/mypath/patients/all_pheno_minus_out.txt', sep=' ')
    withdrawal = pd.read_csv('/mypath/patients/withdrawal.txt', sep=' ')
    eids = not_related[~not_related['f.eid'].isin(withdrawal['f.eid'])]['f.eid'].tolist()
    print('TOTAL NUMBER OF IDS TO SAMPLE FROM: ', len(eids))
    print('not related: ', len(not_related), 'withdrawal: ', len(withdrawal))

    # load data completeness per patient and filter out patients with no genomics:
    compl = pd.read_csv('/mypath/patients/with_snp.tsv', sep='\t')
    #compl = compl[compl['notna']>0.5]
    eids = list(set(compl['f.eid'].to_list()) & set(eids))

    sampled_ids = eids
    merged = merged[merged_columns]
    reduced_merged = merged[merged['f.eid'].isin(sampled_ids)]
    reduced_merged = reduced_merged.rename(columns=rename_dict, inplace=False)

    # Load all genotype files, keep only sampled ids, split them.
    all_genomics = pd.DataFrame({'f.eid': sampled_ids})
    for chrn in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]:
        chrdf = pd.read_csv(f'/mypath/genomics/step2/data/extracted_genotypes/chr{chrn}/chr{chrn}_prep.tsv', sep='\t')
        all_genomics = pd.merge(all_genomics, chrdf, on='f.eid', how='left')
    rename_gen = {f'{all_genomics.columns[i]}':f'{all_genomics.columns[i]}_cluster' for i in range(1,len(all_genomics.columns))}
    all_genomics = all_genomics.rename(columns=rename_gen)


    all_pqtl = pd.DataFrame({'f.eid': sampled_ids})
    for chrn in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]:
        chrdf = pd.read_csv(f'/mypath/genomics/step2/data/pQTL_extracted_genotypes/chr{chrn}/chr{chrn}_prep.tsv', sep='\t')
        all_pqtl = pd.merge(all_pqtl, chrdf, on='f.eid', how='left')
    rename_pqtl = {f'{all_pqtl.columns[i]}':f'{all_pqtl.columns[i]}_pqtl' for i in range(1,len(all_pqtl.columns))}
    all_pqtl = all_pqtl.rename(columns=rename_pqtl)



    ### Calculate missingness per feature
    # reduced_merged, proteomics, genomics, pQTLs

    rm_missingness = pd.DataFrame(data=reduced_merged.columns[1:], columns=['feature'])
    print('number of features in merged: ', len(reduced_merged.columns[1:]))
    rm_missingness['completeness'] = reduced_merged[reduced_merged.columns[1:]].notna().mean().to_numpy()

    prot_missingness = pd.DataFrame(data=prot.columns[1:], columns=['feature'])
    print('number of features in prot: ', len(prot.columns[1:]))
    prot_missingness['completeness'] = prot[prot.columns[1:]].notna().mean().to_numpy()

    gen_missingness = pd.DataFrame(data=all_genomics.columns[1:], columns=['feature'])
    print('number of features in gen: ', len(all_genomics.columns[1:]))
    gen_missingness['completeness'] = all_genomics[all_genomics.columns[1:]].notna().mean().to_numpy()

    pqtl_missingness = pd.DataFrame(data=all_pqtl.columns[1:], columns=['feature'])
    print('number of features in pQTL: ', len(all_pqtl.columns[1:]))
    pqtl_missingness['completeness'] = all_pqtl[all_pqtl.columns[1:]].notna().mean().to_numpy()

    missingness = pd.concat([rm_missingness, prot_missingness, gen_missingness, pqtl_missingness], ignore_index=True)
    missingness.to_csv('/mypath/patients/feature_completeness_prep3.csv', index=False)




if __name__ == "__main__":
    mode = sys.argv[1]
    folder_name = sys.argv[2]
    reduce_proteomics = sys.argv[3]
    try: 
        seed = int(sys.argv[4])
    except:
        seed = None
    print((folder_name, reduce_proteomics, seed))
    if mode == 'completeness_per_patient':
        completeness_per_patient(folder_name, reduce_proteomics, seed)
    elif mode == 'snp':
        get_snp_patients(folder_name, reduce_proteomics, seed)
    elif mode == 'completeness_per_feature':
        completeness_per_feature(folder_name, reduce_proteomics, seed)
    else:
        print('funciton not found')