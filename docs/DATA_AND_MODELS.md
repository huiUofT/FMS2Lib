# Data and models

Article: [FMS2Lib preprint, DOI 10.26434/chemrxiv.15010212/v1](https://doi.org/10.26434/chemrxiv.15010212/v1).

## Bundled assets

| Path | Contents | Source and use |
|---|---|---|
| `data/FMS2Lib/FMS2Lib.mgf` | 63,046 spectra: 34,686 fluorinated and 28,360 non-fluorinated SMILES entries | Project input for dataset preparation and sampling |
| `data/foriceberg/spec_datasets/F_ICEBERG_dataset/labels.tsv` | 63,046 spectrum labels | Spectrum IDs, SMILES, and precursor masses correspond to the full MGF |
| `data/foriceberg/spec_datasets/F_ICEBERG_dataset/spec_files.hdf5` | Spectrum data, distributed with Git LFS | The inventory contains all 63,046 label IDs |
| `data/foriceberg/spec_datasets/F_ICEBERG_dataset/testset_split_1_labels.tsv` | 6,308 test labels | A test-label list; the full train/validation/test assignment is generated separately |
| `data/foriceberg/examples/` | Example spectra, SusDat candidates, formula predictions, and candidate tables | See the [example inventory](../data/foriceberg/examples/README.md) |
| `data/testdataset/` | Formula prediction and neutral-loss examples, including `toxcast_cleaned_norm.mgf` | Example inputs; exact ToxCast release and cleaning provenance are not recorded |
| `data/foriceberg/hf_predict/tag.csv` | 34,686 HF classification rows | Project classifier input |
| `mode/model_ms1_ms2_ranker.joblib` | LightGBM formula ranker | Uses MSBuddy-derived features |
| `mode/foriceberg/generation/best.ckpt` | ICEBERG generation model | Locally retrained checkpoint; records Lightning 2.6.1 |
| `mode/foriceberg/intensity/best.ckpt` | ICEBERG intensity model | Locally retrained checkpoint; records Lightning 2.6.1 |

Original software and documentation use the [MIT License](../LICENSE). Third-party terms and the scope for data and model assets are described in [third-party code and licenses](../README.md#third-party-code-and-licenses). Exact external source releases for some bundled examples and complete original training manifests are not recorded.

## Generated training inputs

Generate these files under `data/foriceberg/spec_datasets/F_ICEBERG_dataset/` before retraining:

| Relative path | Purpose | Generator |
|---|---|---|
| `splits/split_1.tsv` | Train/validation/test assignment | `script/foriceberg/process_new_dataset_from_mgf_with_parentmass.py --seed 1` |
| `subformulae/no_subform.hdf5` | Subformula and intensity tables | `script/foriceberg/assign_subformulae.py` |
| `magma_outputs/magma_tree.hdf5` | Fragmentation trees | `script/foriceberg/run_magma.py`; creates an alias for `magma_tree_new.hdf5` |

Follow [README §5.8](../README.md#58-retrain-f-iceberg). No prepared retraining archive is distributed. Dataset preparation writes labels and spectra as well as the split; use a separate dataset name to preserve an existing dataset. A regenerated split has not been verified against the original checkpoint training split.

## External downloads

| Asset | Source | Local location |
|---|---|---|
| PubChem formula-to-structure database, approximately 6.25 GB | [Zenodo record 15529765](https://zenodo.org/records/15529765/files/pubchem_formulae_inchikey.hdf5), using [`download_pubchem_hdf5.py`](../script/foriceberg/download_pubchem_hdf5.py) | `data/foriceberg/pubchem_formulae_inchikey.hdf5` |
| MSBuddy formula and fragment/loss databases | [Upstream data release v0.2.4](https://github.com/Philipbear/msbuddy/releases/tag/msbuddy_data_v0.2.4) | `src/msbuddy/data/`; downloaded automatically when initialized |
| MSBuddy pretrained ML models | [Upstream data/model release v0.3.0](https://github.com/Philipbear/msbuddy/releases/tag/msbuddy_data_v0.3.0) | Same cache directory; separate from the bundled FMS2Lib ranker |

These external downloads are not included in the repository. The PubChem downloader skips an existing output unless `--force` is supplied; it does not verify an expected checksum. The MSBuddy loader downloads missing files or files below its size threshold and likewise does not check an expected checksum.

## Interpretation and scope

- Counts refer to spectrum records and stored SMILES strings, not an independently normalized count of unique compounds.
- The data splitter stratifies spectrum rows by fluorine count. The bundled test labels and remaining labels have no identical full InChIKeys, but share 14 connectivity blocks (the first InChIKey segment).
- The full MGF lacks collision-energy fields. Preprocessing supplies `30.0`; this is a default, not verified acquisition metadata.
- The PubChem Top100 benchmark uses the true structure to select similar candidates and includes the true answer. It can contain up to 101 candidates per query and is distinct from blind retrieval.
- Small examples demonstrate input formats and execution. See [CPU environments and example outputs](VALIDATION.md) for the supported checks and their limits. GPU execution, full-scale retrieval, training, and article metrics have not been validated.

## Asset checksums

SHA-256 checksums identify the distributed assets. They provide integrity references, not evidence of experimental provenance or model accuracy.

```text
59ef73b90a19b10ee3c76df04e44f9e12d38d4cde0e2e73f326d4348fa862f8f  data/FMS2Lib/FMS2Lib.mgf
77ff499d1c76065f25e08fbf4ed74f1e99c80e64a8a6f34eaf84d9c64be1a22d  data/foriceberg/spec_datasets/F_ICEBERG_dataset/labels.tsv
06ebceffcbdebb40401943cf31f6622e96c2702ec06f800423a793441addb59b  data/foriceberg/spec_datasets/F_ICEBERG_dataset/spec_files.hdf5
cead7588e85c40ba15020cf3c00dbbaf6d2da42cc4b8d22fb1388dd3b0207944  data/foriceberg/spec_datasets/F_ICEBERG_dataset/testset_split_1_labels.tsv
79a7d0f9f6f0f3bbcd33e198314234be004b442d99d035edf5368a9a52ab1d8d  mode/model_ms1_ms2_ranker.joblib
8162a3733b5c9a000fc6aaa26be30be7cc0dab0a4f21b7699fffbc0399d16d25  mode/foriceberg/generation/best.ckpt
637c271f53aea0fa1e41956b41c88b56bbdbcf04ccdfc1d05b884c055fd1347b  mode/foriceberg/intensity/best.ckpt
```
