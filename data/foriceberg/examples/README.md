# ICEBERG example inputs (in git)

| File | Used by |
|------|---------|
| `example_smiles.txt` | `smiles_to_predicted_mgf.py` |
| `mini_onepass_test.mgf` | `run_end2end_from_mgf_susdat_onepass.py` (quick smoke) |
| `F_ICEBERG_calibrated_test10.mgf` | end2end, blind retrieval, retrieval eval |
| `susdat_F.csv` | end2end, blind retrieval |
| `predictions_model_ms1_ms2_ranker.csv` | blind retrieval (pred formulas for test10 MGF) |
| `pubchem_top100_test10.csv` | `run_iceberg_retrieval_eval_evaltest.py` (test10 PubChem top100 candidates) |
| `pool_retrieval_mini.mgf` | `rank_pool_candidates_from_mgf.py` (1 spectrum: MS2Base_172408) |
| `pool_candidates_mini.csv` | `rank_pool_candidates_from_mgf.py` (true + near + far distractor) |
| `F_ICEBERG_train100.mgf` | Optional 100-spectrum smoke input for README §5.8 (seed=1 from `data/FMS2lib/F_ICEBERG.mgf`) |

See README **§5.8 Retrain F-ICEBERG** for the full MGF → MAGMA → gen/inten training commands.

> **Note:** PubChem-by-formula HDF5 (~6 GB) is downloaded separately with `python script/foriceberg/download_pubchem_hdf5.py` to `data/foriceberg/pubchem_formulae_inchikey.hdf5`. Download it before running commands that pass `--pubchem-hdf5` or regenerate `pubchem_top100_test10.csv`.
