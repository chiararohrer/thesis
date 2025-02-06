configfile:"configpath/configname.yaml"

dataset = config["dataset"]
overall_path = config['overallpath']
size = config["size"]
time_cutoff = config["time_cutoff"]

seed = config["seed"]
survival_seed = config["survival_seed"]

with_nores = config["with_nores"]
store_surv_models = config["store_surv_models"]
reduced_run = config["reduced_run"]

run_letter = config["letter"]
global_letter = config["letter"]
res_list = ["res", "nores"] if with_nores else ["res"]
omics = config["omics"]

wildcard_constraints:
    datasets = "prep3",
    dataset = "prep3",
    res = "res",
    size = "52026",
    run_letters = "T",
    omics2 = "[a-zA-Z_]+"


rule all:
    input: 
        expand("{overall_paths}/baseline/print_{datasets}_{res}_{sizes}_{run_letters}_{omics2}/results_print_{datasets}_{res}_{sizes}_{run_letters}_{omics2}_{time_cutoffs}.tsv", res=res_list, omics2=omics, overall_paths=[overall_path], datasets=[dataset], sizes=[size], time_cutoffs=[time_cutoff], run_letters=[run_letter])
    threads: 1

        
rule survival:
    input:
        result_folder = "{overall_path}/{dataset}/results_res_400000_A/latent_space_{omics2}",
        interim_folder = "{overall_path}/{dataset}/interim_data_{res}_{sizes}_{run_letters}"
    output:
        "{overall_path}/baseline/print_{dataset}_{res}_{sizes}_{run_letters}_{omics2}/results_print_{dataset}_{res}_{sizes}_{run_letters}_{omics2}_{time_cutoffs}.tsv"
    params:
        res_temp = "{res}"
    threads: 2
    shell:
        """
        echo "result_folder={input.result_folder}"
        echo "interim_folder={input.interim_folder}"
        cd /mypath/python_scripts 
        if [ "{survival_seed}" == "True" ] && [ "{reduced_run}" == "False" ]; then
            python transfer_wrapper.py "all" {dataset} "results_res_400000_A/latent_space_{wildcards.omics2}" "interim_data_{params.res_temp}_{size}_{run_letter}" "id_{size}" "/mypath/baseline/print_{dataset}_{params.res_temp}_{size}_{run_letter}_{wildcards.omics2}" {time_cutoff} {store_surv_models} {seed} "{wildcards.omics2}_{size}.pt"
        elif [ "{survival_seed}" == "True" ] && [ "{reduced_run}" == "True" ]; then
            python transfer_wrapper.py "reduced" {dataset} "results_res_400000_A/latent_space_{wildcards.omics2}" "interim_data_{params.res_temp}_{size}_{run_letter}" "id_{size}" "/mypath/baseline/print_{dataset}_{params.res_temp}_{size}_{run_letter}_{wildcards.omics2}" {time_cutoff} {store_surv_models} {seed} "{wildcards.omics2}_{size}.pt"
        elif [ "{survival_seed}" == "False" ] && [ "{reduced_run}" == "True" ]; then
            python transfer_wrapper.py "reduced" {dataset} "results_res_400000_A/latent_space_{wildcards.omics2}" "interim_data_{params.res_temp}_{size}_{run_letter}" "id_{size}" "/mypath/baseline/print_{dataset}_{params.res_temp}_{size}_{run_letter}_{wildcards.omics2}" {time_cutoff} {store_surv_models} "None" "{wildcards.omics2}_{size}.pt"
        else
            python transfer_wrapper.py "all" {dataset} "results_res_400000_A/latent_space_{wildcards.omics2}" "interim_data_{params.res_temp}_{size}_{run_letter}" "id_{size}" "/mypath/baseline/print_{dataset}_{params.res_temp}_{size}_{run_letter}_{wildcards.omics2}" {time_cutoff} {store_surv_models} "None" "{wildcards.omics2}_{size}.pt"
        fi
        """

