# ICEBERG example inputs (in git)

| File | Used by |
|------|---------|
| `example_smiles.txt` | `smiles_to_predicted_mgf.py` |
| `mini_onepass_test.mgf` | `run_end2end_from_mgf_susdat_onepass.py` (quick smoke) |
| `F_ICEBERG_calibrated_test10.mgf` | end2end, blind retrieval, retrieval eval |
| `susdat_F.csv` | end2end, blind retrieval |
| `predictions_model_ms1_ms2_ranker.csv` | blind retrieval (pred formulas for test10 MGF) |
| `pubchem_top100_test10_mini.csv` | `run_iceberg_retrieval_eval_evaltest.py` (3 spectra, rank-1 only) |

> **Note:** PubChem-by-formula HDF5 (~6 GB) for `build_pubchem_top100_by_ecfp4_from_mgf.py` and optional blind-retrieval `--pubchem-hdf5` is **not** bundled here. Source / download / packaging — **TBD**.
