#### Overview of pyhton scripts used for the project:
_ captum_script.py: IntegratedGradients for interpretability of VAE encoder & decoder.
_ completeness_calculation.py: Functions to calculate data completeness per feature or per patient.
_ disease_shap_analysis.py: Explainability of survival models on the latent space.
_ icd10_chapters.py: Perprocessing of diagnosis dates per ICD10 chapter. No data on self-reported cases.
_ icd10_prep_all.py: Preprocessing of all ICD10 codes (two digits after letter) if at least 1000 occurrences. No data on self-reported cases.
_ icd10_prep.py: Preprocessing of the 15 selected diseases.
_ knn_clustering.py: Analysis of clustering by categorical covariates in the latent space.
_ move_imputer.py: Analysis of imputation quality using pretrained MOVE VAE.

* prep3: Residualized continuous features, categorical and non-residualized SNPs. Used for majority of analysis.
* prep4: Residualized continuous features, continuous and residualized SNPs. Used for comparison only.
    _ prepX_prep.py: Prepocessing of UK Biobank data for MOVE.
    _ prepX_residualization.py: Residualize previously prepocessed data.
    _ prepX_transfer_prep.py: Preprocessing for the transfer learning ("single-omics") experiment.

_ preprocessing_functions.py: Collections of functions used by other scripts.
_ transfer_wrapper.py: Wrapper for survival models on single-omics data.
_ wrapper_all.py: Wrapper for survival models on all ICD10 codes with at least 1000 occurrences.
_ wrapper.py: Wrapper for survival models.