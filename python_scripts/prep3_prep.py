import numpy as np
import pandas as pd
import random
import sys
from preprocessing_functions import get_id_from_tsv, sample_store


def prep(folder_name, sample_num, reduce_proteomics, letter, seed=None):
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

    #id_visit_date = pd.read_csv('/mypath/diseases/visit_date_code.tsv', sep='\t')

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

    olink_batch_plate = [str(id_batch[id_batch['name']=='Plate used for sample run (Olink)']['id'].values[0])]
    olink_batch_plate = ['f.eid']+['f.' + col + '.0.0' for col in olink_batch_plate]

    olink_batch_well = [str(id_batch[id_batch['name']=='Well used for sample run (Olink)']['id'].values[0])]
    olink_batch_well = ['f.eid']+['f.' + col + '.0.0' for col in olink_batch_well]

    #visit_date = [str(id_visit_date[id_visit_date['name']=='Date of attending assessment centre']['id'].values[0])]
    #visit_date = ['f.eid']+['f.' + col + '.0.0' for col in visit_date]

    merged_columns = ['f.eid'] + age[1:] + fasting[1:] + sex[1:] + assessment_centre[1:] + ethnic_bg[1:] + physical_measurments[1:] + smoking[1:] + alcohol[1:] + blood_count[1:] + blood_biochem[1:] + infect[1:] + metabolomics[1:] + urine[1:] + olink_batch_plate[1:] + olink_batch_well[1:]  #+ visit_date[1:]


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
    obp_df = merged[olink_batch_plate]
    obp_df = obp_df.rename(columns=rename_dict, inplace=False)
    obw_df = merged[olink_batch_well]
    obw_df = obw_df.rename(columns=rename_dict, inplace=False)
    

    #vd_df = merged[visit_date]
    #vd_df.to_csv(f'/mypath/diseases/visit_date.tsv', sep='\t', index=False)

    prot = pd.read_csv(olink_path, sep='\t', compression='gzip')
    prot = prot[prot['ins_index'] == 0]
    prot = prot.pivot(index='eid', columns='protein_id', values='result')
    prot.reset_index(inplace=True)
    prot.columns.name = None

    # merge proteomics with sex to get all ids into proteomics and then only retain proteomic columns
    prot_allids = pd.merge(prot, sex_df, left_on='eid', right_on='f.eid', how='right')
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
    eids = list(set(compl['f.eid'].to_list()) & set(eids))

    print('TOTAL NUMBER OF IDS TO SAMPLE FROM: ', len(eids))
    sampled_ids = random.sample(eids, sample_num)



    # TODO: Load all genotype files, keep only sampled ids, split them.
    all_genomics = pd.DataFrame({'f.eid': sampled_ids})
    for chrn in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]:
        chrdf = pd.read_csv(f'/mypath/genomics/step2/data/extracted_genotypes/chr{chrn}/chr{chrn}_prep.tsv', sep='\t')
        all_genomics = pd.merge(all_genomics, chrdf, on='f.eid', how='left')
    rename_gen = {f'{all_genomics.columns[i]}':f'{all_genomics.columns[i]}_cluster' for i in range(1,len(all_genomics.columns))}
    all_genomics = all_genomics.rename(columns=rename_gen)
        
    sample_store(folder_name, 'genomics', all_genomics, sampled_ids, letter)

    all_pqtl = pd.DataFrame({'f.eid': sampled_ids})
    for chrn in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]:
        chrdf = pd.read_csv(f'/mypath/genomics/step2/data/pQTL_extracted_genotypes/chr{chrn}/chr{chrn}_prep.tsv', sep='\t')
        all_pqtl = pd.merge(all_pqtl, chrdf, on='f.eid', how='left')
    rename_pqtl = {f'{all_pqtl.columns[i]}':f'{all_pqtl.columns[i]}_pqtl' for i in range(1,len(all_pqtl.columns))}
    all_pqtl = all_pqtl.rename(columns=rename_pqtl)

    sample_store(folder_name, 'pQTL', all_pqtl, sampled_ids, letter)

    

    '''
    Apply the function sample_store on all subsets. This function filters the dataframes for the sampled eIDs and stores the dataframes as tsv tables in the defined folder with the defined name.
    Then, the sampled eID subset is stored as a txt file, as required by MOVE.
    '''
    sample_store(folder_name, 'age', age_df, sampled_ids, letter)
    sample_store(folder_name, 'fasting_time', fasting_df, sampled_ids, letter)
    sample_store(folder_name, 'sex', sex_df, sampled_ids, letter)
    sample_store(folder_name, 'assessment_centre', ac_df, sampled_ids, letter)
    sample_store(folder_name, 'ethnic_background', ebg_df, sampled_ids, letter)
    sample_store(folder_name, 'smoking_status', sm_df, sampled_ids, letter)
    sample_store(folder_name, 'alcohol_frequency', al_df, sampled_ids, letter)
    sample_store(folder_name, 'physical_measurements', pm_df, sampled_ids, letter)
    sample_store(folder_name, 'blood_count', bc_df, sampled_ids, letter)
    sample_store(folder_name, 'blood_biochemistry', bb_df, sampled_ids, letter)
    sample_store(folder_name, 'infectious_diseases', inf_df, sampled_ids, letter)
    sample_store(folder_name, 'urine', urine_df, sampled_ids, letter)
    sample_store(folder_name, 'metabolomics', met_df, sampled_ids, letter)
    sample_store(folder_name, 'olink_batch_plate', obp_df, sampled_ids, letter)
    sample_store(folder_name, 'olink_batch_well', obw_df, sampled_ids, letter)
    sample_store(folder_name, 'proteomics', prot, sampled_ids, letter)

    id_df = pd.DataFrame({'f.eid': sampled_ids})
    id_df.to_csv(f'/mypath/{folder_name}/data_{letter}/id_{sample_num}.txt', sep=' ', index=False, header=False)


if __name__ == "__main__":
    folder_name = sys.argv[1]
    sample_num = int(sys.argv[2])
    reduce_proteomics = sys.argv[3]
    letter = sys.argv[4]
    try: 
        seed = int(sys.argv[5])
    except:
        seed = None
    print((folder_name, sample_num, reduce_proteomics, letter, seed))
    prep(folder_name, sample_num, reduce_proteomics, letter, seed)