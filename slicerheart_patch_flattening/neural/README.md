# neural

Learned surrogates trained on the synthetic baffle dataset (`DATASET.md`, release `dataset-v1`).

- `data.py`: featurisation (canonical frame, arclength resampling, surface sampling, residual target)
- `dataset_io.py`: shard loader, deterministic 80/10/10 split by sample id, `BaffleDataset` for PyTorch
- `model.py`: `OutlineSurrogate` (circular 1D CNN boundary encoder + PointNet surface encoder)
- `train.py`: training, baselines, scaling study; `--device cuda --amp --shift-invariant --shards ...`
- `data/sample_100/`: 100-sample slice of the dataset for tests and format reference

Feasibility study: `reports/04_neural_surrogate_feasibility.md`. Training path and roadmap to the
inverse (pressurised) surrogate: `reports/05_inverse_surrogate_roadmap.md`.
