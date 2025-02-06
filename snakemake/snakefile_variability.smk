configfile:"configpath/configname.yaml"

dataset = config["dataset"]
overall_path = config['overallpath']
size = config["size"]
time_cutoff = config["time_cutoff"]

seed = config["seed"]
preprocessing_seed = config["preprocessing_seed"]
encode_seed = config["encode_seed"]
latent_seed = config["latent_seed"]
survival_seed = config["survival_seed"]

with_nores = config["with_nores"]
store_surv_models = config["store_surv_models"]
reduced_run = config["reduced_run"]

one_latent = config["one_latent"]
reduce_proteomics = config["reduce_proteomics"]


run_letter = config["letter"]
global_letter = config["letter"]

res_list = ["res", "nores"] if with_nores else ["res"]

rule all:
    input:
        # update: folder with pdfs and csv from survival -> change this to last written file
        expand("{overall_paths}/baseline/{datasets}_{res}_{sizes}_{run_letters}{runs}/results_{datasets}_{res}_{sizes}_{run_letters}{runs}_{time_cutoffs}.tsv", res=res_list, runs=[f'{i}' for i in range(config['variability_runs'])], overall_paths=[overall_path], datasets=[dataset], sizes=[size], time_cutoffs=[time_cutoff], run_letters=[run_letter])
    threads: 1


def get_config_paths():
    if not one_latent:
        return "{overall_path}/{dataset}/config/data/{dataset}_nores_{size}_{run_letters}{runs}.yaml", "{overall_path}/{dataset}/config/data/{dataset}_res_{size}_{run_letters}{runs}.yaml"
    else:
        return "{overall_path}/{dataset}/config/data/{dataset}_nores_{size}_{run_letter}.yaml", "{overall_path}/{dataset}/config/data/{dataset}_res_{size}_{run_letter}.yaml"

rule make_data_config:
    input:
        "{overall_path}/{dataset}/config/data/{dataset}_nores_template.yaml",
        "{overall_path}/{dataset}/config/data/{dataset}_res_template.yaml"
    output:
        get_config_paths()
    params: 
        temp_runs = lambda wildcards: "None" if one_latent else f"{wildcards.run_letters}{wildcards.runs}"
    
    threads: 1
    shell:
        """
        cd /mypath/python_scripts/
        if [ "{one_latent}" == "True" ]; then
            python -c "from preprocessing_functions import data_yaml_from_template; data_yaml_from_template('{overall_path}/{dataset}/config/data/{dataset}_nores_template.yaml', {size}, 'None', '{run_letter}', 'False', 'False')"
            python -c "from preprocessing_functions import data_yaml_from_template; data_yaml_from_template('{overall_path}/{dataset}/config/data/{dataset}_res_template.yaml', {size}, 'None', '{run_letter}', 'False', 'False')"
        else
            python -c "from preprocessing_functions import data_yaml_from_template; data_yaml_from_template('{overall_path}/{dataset}/config/data/{dataset}_nores_template.yaml', {size}, '{params.temp_runs}', '{run_letter}', 'False', 'True')"
            python -c "from preprocessing_functions import data_yaml_from_template; data_yaml_from_template('{overall_path}/{dataset}/config/data/{dataset}_res_template.yaml', {size}, '{params.temp_runs}', '{run_letter}', 'False', 'True')"
        fi
        """


rule preprocess:
    input:
        "{overall_path}/{dataset}/preprocessing"
    output: 
        "{overall_path}/{dataset}/data_{run_letters}/id_{size}.txt"
    threads: 6
    resources:
        mem_mb = 400000
    shell:
        """
        cd {overall_path}/python_scripts
        echo "{reduce_proteomics}"
        if [ "{preprocessing_seed}" == "True" ]; then
            python {dataset}_prep.py "{dataset}" "{size}" "{reduce_proteomics}" "{wildcards.run_letters}" "{seed}"
        else
            python {dataset}_prep.py "{dataset}" "{size}" "{reduce_proteomics}" "{wildcards.run_letters}"
        fi
        """


rule encode:
    input: 
        # preprocessed tsv data
        data="{overall_path}/{dataset}/data_{run_letters}/id_{size}.txt"
    params:
        config_name = lambda wildcards: f"{wildcards.dataset}_nores_{wildcards.size}_{wildcards.run_letters}0.yaml" if not one_latent else f"{wildcards.dataset}_nores_{wildcards.size}_{wildcards.run_letters}.yaml"
    output:
        directory("{overall_path}/{dataset}/interim_data_nores_{size}_{run_letters}")

    threads: 1
    resources:
        mem_mb_per_cpu=100000
    shell:
        # adapt config (data)
        """
        cd {overall_path}/{dataset}
        if [ "{encode_seed}" == "True" ]; then
            move-dl data={params.config_name} task=encode_data seed={seed}
        else
            move-dl data={params.config_name} task=encode_data
        fi
        """


rule residualize:
    input: 
        # encoded data
        "{overall_path}/{dataset}/interim_data_nores_{size}_{run_letters}"
    output: 
        # residualized interim data
        directory("{overall_path}/{dataset}/interim_data_res_{size}_{run_letters}")
    params:
        temp_runs= lambda wildcards: f"{wildcards.run_letters}" if not one_latent else {wildcards.run_letters}
    threads: 2
    resources:
        mem_mb_per_cpu=100000 
    shell:
        """
        cd /mypath/python_scripts
        python {dataset}_residualization.py "{overall_path}/{dataset}" "{dataset}_nores_{size}_{params.temp_runs}" "{size}" "{params.temp_runs}"
        """


def get_result_path():
    if not one_latent:
        return directory("{overall_path}/{dataset}/results_{res}_{size}_{run_letters}{runs}/latent_space")
    else:
        return directory("{overall_path}/{dataset}/results_{res}_{size}_{run_letters}/latent_space")


rule latent_analysis:
    input:
        interim="{overall_path}/{dataset}/interim_data_{res}_{size}_{run_letters}",
        data_config = lambda wildcards: "{overall_path}/{dataset}/config/data/{dataset}_{res}_{size}_{run_letters}{runs}.yaml" if not one_latent else "{overall_path}/{dataset}/config/data/{dataset}_{res}_{size}_{run_letters}.yaml"
    params:
        res_temp = "{res}",
        runs_temp = lambda wildcards: f"{wildcards.run_letters}{wildcards.runs}" if not one_latent else {wildcards.run_letters}
    output: 
        get_result_path()
    threads: 1
    resources:
        # Run this task on the GPU queue
        slurm_partition="gpuqueue",
        # Reserve 1 GPU for this job
        slurm_extra="--gres=gpu:1",
        mem_mb_per_cpu=100000 
    shell:
        """
        cd {overall_path}/{dataset}
        if [ "{latent_seed}" == "True" ]; then
            HYDRA_FULL_ERROR=1 move-dl data={dataset}_{params.res_temp}_{size}_{params.runs_temp} task=analyze_latent_{dataset}_cuda seed={seed}
        else
            HYDRA_FULL_ERROR=1 move-dl data={dataset}_{params.res_temp}_{size}_{params.runs_temp} task=analyze_latent_{dataset}_cuda
        fi
        """

        
rule survival:
    input:
        result_folder = lambda wildcards: "{overall_path}/{dataset}/results_{res}_{size}_{run_letters}{runs}/latent_space" if not one_latent else "{overall_path}/{dataset}/results_{res}_{size}_{run_letter}/latent_space",
        interim_folder = "{overall_path}/{dataset}/interim_data_{res}_{size}_{run_letter}"
    output:
        "{overall_path}/baseline/{dataset}_{res}_{size}_{run_letters}{runs}/results_{dataset}_{res}_{size}_{run_letter}{runs}_{time_cutoff}.tsv"
    params:
        res_temp = "{res}",
        runs_temp = lambda wildcards: f"{wildcards.run_letters}{wildcards.runs}" if not one_latent else {wildcards.run_letters}
    threads: 4
    resources:
        mem_mb_per_cpu=150000
    shell:
        """
        cd /mypath/python_scripts 
        if [ "{survival_seed}" == "True" ] && [ "{reduced_run}" == "False" ]; then
            python wrapper.py "all" {dataset} "results_{params.res_temp}_{size}_{params.runs_temp}" "interim_data_{params.res_temp}_{size}_{run_letter}" "id_{size}" "/mypath/baseline/{dataset}_{params.res_temp}_{size}_{run_letter}{wildcards.runs}" {time_cutoff} {store_surv_models} {seed} "{wildcards.run_letters}"
        elif [ "{survival_seed}" == "True" ] && [ "{reduced_run}" == "True" ]; then
            python wrapper.py "reduced" {dataset} "results_{params.res_temp}_{size}_{params.runs_temp}" "interim_data_{params.res_temp}_{size}_{run_letter}" "id_{size}" "mypath/baseline/{dataset}_{params.res_temp}_{size}_{run_letter}{wildcards.runs}" {time_cutoff} {store_surv_models} {seed} "{wildcards.run_letters}"
        elif [ "{survival_seed}" == "False" ] && [ "{reduced_run}" == "True" ]; then
            python wrapper.py "reduced" {dataset} "results_{params.res_temp}_{size}_{params.runs_temp}" "interim_data_{params.res_temp}_{size}_{run_letter}" "id_{size}" "/mypath/baseline/{dataset}_{params.res_temp}_{size}_{run_letter}{wildcards.runs}" {time_cutoff} {store_surv_models} "None" "{wildcards.run_letters}"
        else
            python wrapper.py "all" {dataset} "results_{params.res_temp}_{size}_{params.runs_temp}" "interim_data_{params.res_temp}_{size}_{run_letter}" "id_{size}" "/mypath/baseline/{dataset}_{params.res_temp}_{size}_{run_letter}{wildcards.runs}" {time_cutoff} {store_surv_models} "None" "{wildcards.run_letters}"
        fi
        """