# CPU environments and example outputs

Python 3.11 was used for the small CPU examples below. Formula prediction and structure annotation used separate environments. These examples do not establish GPU compatibility, full retraining reproducibility, or article benchmark results.

## Environment records

| Component | Formula prediction | ICEBERG structure annotation |
|---|---|---|
| NumPy | 1.26.4 | 1.26.4 |
| LightGBM | 4.5.0 | — |
| scikit-learn | 1.5.2 | — |
| PyTorch | — | 2.5.1+cpu |
| DGL | — | 1.1.3 |
| torch-scatter | — | 2.1.2+pt25cpu |
| torch-sparse | — | 0.6.18+pt25cpu |
| PyTorch Geometric | — | 2.8.1 |
| PyTorch Lightning | — | 1.9.5 |
| RDKit | — | 2024.03.5 |
| PyArrow | — | 19.0.1 |
| Ray | — | 2.58.0 |
| pygmtools | — | 0.6.0 |

The table lists base versions; some installed distributions used cluster-specific `+computecanada` builds. It is a compatibility reference, not a portable lock file or a freshly verified installation recipe. On Alliance clusters, the structure environment used `arrow/19.0.1` and `rdkit/2024.03.5` modules. Other systems should provide compatible packages through their own installation method.

`requirements.txt` remains a general dependency specification. Match the native PyG extensions to the selected PyTorch build. The CPU examples used DGL 1.1.3; newer DGL distributions may constrain PyTorch to other versions.

## Example outputs

| Example | Expected output |
|---|---|
| Ten-spectrum formula inference | Ten predictions matching `data/foriceberg/examples/predictions_model_ms1_ms2_ranker.csv`; 146 candidate feature rows |
| Fixed pool: one spectrum, three candidates | The `query_true` candidate ranks first, with similarity approximately 0.9472 |
| Ten-spectrum formula-to-structure integration, capped at three candidates per query | Ten annotation rows and thirty candidate scores using PubChem objects for the predicted formulas |

Reference predictions are model outputs, not guaranteed correct labels. Eight of the ten reference formulas match the MGF `FORMULA` fields. The three-candidate pool is an example, not a representative retrieval benchmark.

The integration example was checked with complete PubChem objects for the ten predicted formulas and `--max-candidates 3`. The full database file and the default 400-candidate configuration were not evaluated. For ordinary use, obtain the [external PubChem database](DATA_AND_MODELS.md#external-downloads) and follow the README workflow.

## CPU example commands

Run from the repository root with the source paths in README §1 configured. Use the corresponding environment for each stage.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
python script/formula_prediction/eval/run_eval_generate_and_predict.py \
  --mgf data/foriceberg/examples/F_ICEBERG_calibrated_test10.mgf \
  --mgf-only --output-dir scratch/formula_test10 \
  --model mode/model_ms1_ms2_ranker.joblib --no-mmap --n-cpu 1
```

```bash
DGLBACKEND=pytorch OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python script/foriceberg/rank_pool_candidates_from_mgf.py \
  --mgf data/foriceberg/examples/pool_retrieval_mini.mgf \
  --candidates data/foriceberg/examples/pool_candidates_mini.csv \
  --out-dir scratch/pool_retrieval_mini
```

## Compatibility and output notes

- LightGBM 4.5.0 and scikit-learn 1.8.0 have an incompatible `check_array` interface. The formula example used scikit-learn 1.5.2.
- The checkpoints record Lightning 2.6.1. The CPU examples loaded them with Lightning 1.9.5, which emits a version warning. Compatibility with training or other inputs is not established.
- The bundled SusDat table has no candidates for the ten reference predicted formulas. Use PubChem for that example. If all queries lack candidates, blind retrieval writes `skipped_spectra.csv` and returns exit status 0 without an annotation CSV; inspect the output files as well as the exit status.
