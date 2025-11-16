# grid

## Training

Run experiments through the Lightning + Hydra entrypoint:

```bash
python train.py
```

Override any config value from `configs/train.yaml`, e.g.:

```bash
python train.py trainer.max_epochs=1 data.case_name=pglib_opf_case30_ieee
```

The script automatically computes feature statistics with `OPFNormalizer` on the
training split (saved to `normalizer.stats_path`) before fitting the model.
Existing stats are reused unless `normalizer.overwrite=true` is provided.
