# Partial-layer IntenGNN fine-tune (FMS2Lib)

See `finetune_inten_partial_layers.py --help`. Data under `data/foriceberg/spec_datasets/`, weights under `mode/foriceberg/`.

```bash
cd /path/to/FMS2Lib
python3 script/foriceberg/finetune_inten_partial_layers.py \
  --pretrained-checkpoint mode/foriceberg/intensity/best.ckpt \
  --list-modules
```

**Only Transformer (before MLP):** `--preset before_mlp`

**MLP + output head:** `--preset mlp_head`

**Custom:** `--train-modules trans_layers,intermediate_out`

Requires `--magma-dag-folder` (same as full inten retrain).
