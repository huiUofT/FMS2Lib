# Data for portable `github/` workflows

Training scripts under `github/sh/` and `github/script/run_retrain_ficeberg_split1_newloc.py` expect the **F_ICEBERG** layout **inside this `github/` tree**:

```text
github/
  data/
    spec_datasets/
      F_ICEBERG_dataset/
        labels.tsv
        spec_files.hdf5
        splits/split_1.tsv
        magma_outputs/magma_tree.hdf5   (or magma_tree_new.hdf5 + link)
        subformulae/no_subform.hdf5
```

## Populate (on your machine, before `git push`)

From the parent ICEBERG checkout (one level above `github/`), you can symlink without copying huge files:

```bash
GITHUB=/path/to/iceberg/github
mkdir -p "${GITHUB}/data/spec_datasets"
ln -sfn /path/to/iceberg/data/spec_datasets/F_ICEBERG_dataset \
  "${GITHUB}/data/spec_datasets/F_ICEBERG_dataset"
```

Or `rsync` / copy the directory if you want a fully self-contained clone for GitHub LFS / release assets.

**Do not** rely on `github/database/` symlinks pointing outside `github/` if you need the subtree to be relocatable; use `data/spec_datasets/` as above.

For a hard check that dataset files **resolve** under `github/` (no symlink targets outside the tree), run retrain with **`ICEBERG_STRICT_GITHUB=1`** (see `sh/README.md`).
