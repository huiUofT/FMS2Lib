# `github/` 脚本说明

本目录可 **单独拷贝/上传到 GitHub**：脚本与 **`github/src/ms_pred`** 自洽；大体积数据请放在 **`github/data/spec_datasets/`**（见 **`data/README.md`**），或在本机用符号链接挂到该路径。

**Python 依赖路径**（与 `sh/`、`script/` 中编排一致）：

```bash
export PYTHONPATH="/path/to/this/github/src:${PYTHONPATH:-}"
```

若本目录仍嵌在 ICEBERG 大仓库里，也可在 **`github/`** 根下执行，不必再依赖上级 `iceberg/src`。

**依赖清单（pip）**：根目录 **`requirements.txt`** 与仓库根 **`activate_env.sh`** 里的自检、`retrain` 流水线所需包对齐；GPU 栈（torch / dgl / PyG）请按该文件顶部注释顺序安装，或继续用 **`source …/activate_env.sh`**。

**旧式快捷链接**：`database/` 里可能仍有指向上级仓库的 symlink，**不适合随 `github/` 单独上传**；上传请改用 **`data/`** 布局。

---

## `script/process_new_dataset_from_mgf_with_parentmass.py`

**作用**：从 **MGF + 内嵌字段**（`IDENTIFIER`、`SMILES`、`PRECURSOR_MZ` 等）构建 `data/spec_datasets/<name>/` 下的 **`labels.tsv`**、**`spec_files.hdf5`**、**`splits/split_<seed>.tsv`**。在每条谱的 SIRIUS 风格字符串中写入 **`#parentmass`**（紧跟 `#PRECURSORMZ`，数值与该行一致），与官方 **`F_ICEBERG_dataset`** 的 `spec_files` 载荷对齐。

**依赖**：**`github/src`**（脚本自动 `PYTHONPATH` + `chdir` 到 **`github/`** 根），输出写到 **`github/data/spec_datasets/<name>/`**。另需 RDKit、pandas、tqdm。

**示例**（在 **`github/`** 根执行）：

```bash
cd /path/to/FMS2Lib
python3 script/process_new_dataset_from_mgf_with_parentmass.py \
  --mgf-file database/F_ICEBERG.mgf \
  --dataset-name F_ICEBERG_dataset_repro_with_pm \
  --seed 1
```

**说明**：根目录下的 `github/process_new_dataset_from_mgf_with_parentmass.py` 为**转发入口**，会 exec 到 **`script/`** 内实现。英文说明见 **`script/README_process_new_dataset_from_mgf_with_parentmass.md`**。

**主要参数**：`--mgf-file`、`--dataset-name`、`--train-frac` / `--val-frac` / `--test-frac`（默认 0.8/0.1/0.1）、`--seed`（决定 `split_<seed>.tsv` 文件名）。

---

## `script/run_end2end_from_mgf_susdat_onepass.py`

**作用**：**一条命令**串联：MGF（含 `TITLE=`、`FORMULA=` 等）+ SusDat 类 CSV（`formula`/`Name`/`SMILES`）→ 候选构造 → 调用端到端预测脚本 → 余弦等评估；输出写到 `--out-dir`。

**依赖与目录**：入口脚本在 **`github/script/`**；捆绑的 **`ms_pred`** 在 **`github/src/ms_pred/`**；默认 **GEN / INTEN 权重** 在 **`github/mode/generation/best.ckpt`** 与 **`github/mode/intensity/best.ckpt`**（可用 `--gen-checkpoint` / `--inten-checkpoint` 覆盖）。详见 **`github/script/README.md`**。仍需本机安装 PyTorch / DGL 等与模型一致的依赖。

**示例**：

```bash
cd <ICEBERG_REPO_ROOT>

python github/script/run_end2end_from_mgf_susdat_onepass.py \
  --mgf /path/to/input.mgf \
  --susdat /path/to/susdat_F.csv \
  --out-dir /path/to/run_out
```

**主要参数**：`--mgf`、`--susdat`、`--out-dir` 必填；`--gen-checkpoint` / `--inten-checkpoint` 可选（默认见上）。另有可选 `--predict-script`。预测相关参数（`--batch-size` 等）不写时默认 **32 / 0 / 100 / 5000 / 0.05**（见 `github/script/README.md`）。

---

## `run_iceberg_retrieval_eval_evaltest.py`

**作用**：与 **`ms_pred/retrieval/run_iceberg_retrieval_eval.py`**（`github/src` 内）同一条检索管线（`base.parse_args()` 等），但打分使用 **`evaluate_test_set.py` 风格的 dist-bin 余弦**；除 `top1_summary.csv` 外，还写 **逐谱** 的 `per_spectrum_retrieval.tsv`。

**依赖**：运行 **`github/script/run_iceberg_retrieval_eval_evaltest.py`** 或 **`github/run_iceberg_retrieval_eval_evaltest.py`** 时会自动把 **`github/src`** 加入 `sys.path`；另需 `h5py`、`numpy`、`pandas`、`RDKit`。仓库 **`retraiveal/`** 下仍有同名脚本副本，可与本入口二选一。

**示例**：

```bash
cd /path/to/FMS2Lib
python github/script/run_iceberg_retrieval_eval_evaltest.py  # 参数见 ms_pred.retrieval.run_iceberg_retrieval_eval.parse_args()
```

---

## `script/smiles_to_predicted_mgf.py`

**Purpose:** predict **MS/MS** from **SMILES** (single string or file) with **ICEBERG generation + intensity checkpoints** and write **MGF**.

**Paths:** with the script under **`github/script/`**, the logical root is **`github/`**; default weights live under **`github/mode/`**; the subprocess runs **`github/src/ms_pred/dag_pred/predict_smis_trained_inten.py`** with **`PYTHONPATH`** including **`github/src`**. See **`github/script/README_smiles_to_predicted_mgf.md`**.

**Example** (I/O under `github/database/`; set `GITHUB` to your `github` directory):

```bash
GITHUB="<path-to-github-directory>"

python "${GITHUB}/script/smiles_to_predicted_mgf.py" \
  --smiles "CC(=O)O" \
  --out "${GITHUB}/database/pred_single.mgf"
```

**Main flags:** `--smiles` or `--smiles-file`, `--out` (required); `--gen-checkpoint`, `--inten-checkpoint` (default `mode/.../best.ckpt`); optional `--gpu`, `--identifier-prefix`, etc. (see **`README_smiles_to_predicted_mgf.md`** and `argparse` in the script).

---

## Relation to `predict/`

- `github/script/smiles_to_predicted_mgf.py` mirrors `predict/smiles_to_predicted_mgf.py` for a portable **`github/`** tree. When in doubt, compare with the **`predict/`** copy in the full repository.
