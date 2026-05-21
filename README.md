# Kaggle ISLR Reproduction

Current stage: Kaggle ISLR Dataset / DataLoader pipeline.

This repository checks the dataset, creates participant-based folds, reads parquet landmark files, verifies PyTorch `Dataset` / `DataLoader` batches, and includes a tiny smoke-training baseline. The smoke training is only a short pipeline check, not formal model training.

## Data

Default data path:

```powershell
C:\ASL\asl-signs
```

The large Kaggle dataset is not copied into this project. Scripts read it through `--data-root`, which defaults to `C:\ASL\asl-signs`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install pandas numpy pyarrow scikit-learn tqdm
```

Install GPU PyTorch separately with the official CUDA 11.8 wheel index:

```powershell
python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```

PyTorch CUDA wheels require a separate `--index-url`, so `torch`, `torchvision`, and `torchaudio` are intentionally not listed in `requirements.txt`.

Environment check:

```powershell
python -c "import sys, torch; print('python:', sys.version); print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('torch cuda:', torch.version.cuda); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

## Commands

```powershell
python scripts/check_dataset.py
python scripts/inspect_sample.py --index 0
python src/preprocess.py --index 0
python src/split.py --data-root C:\ASL\asl-signs --n-splits 5
python scripts/check_dataloader.py --data-root C:\ASL\asl-signs --fold 0 --batch-size 8
python scripts/check_first_place_preprocess.py --data-root C:\ASL\asl-signs --fold 0 --batch-size 8
python -m unittest tests.test_model_tiny -v
python src/train_smoke.py --data-root C:\ASL\asl-signs --fold 0 --batch-size 16 --max-train-batches 20 --max-valid-batches 5 --epochs 1
python src/train_baseline.py --config configs/tiny_baseline.json
```

`configs/tiny_baseline.json` is currently a small trial-training config, not a full formal training run. It defaults to `epochs=1`, `max_train_batches=100`, and `max_valid_batches=20` so you can verify that train loss, valid loss, valid accuracy, and best/last checkpoint saving work before committing to a full fold run.

## Feature Cache

Online first-place preprocessing reads parquet files and computes `[64, 708]` features on the fly. For faster repeated experiments, you can manually build a `.npy` feature cache.

Small cache build test:

```powershell
python scripts\build_feature_cache.py --csv outputs\first_place_train_fold0.csv --cache-dir C:\ASL\islr_feature_cache_fp16 --max-samples 1000
```

This writes feature files named `{participant_id}_{sequence_id}.npy` and a metadata table at `outputs/cache_metadata.csv`.

Check cached DataLoader:

```powershell
python scripts\check_cached_dataloader.py --csv outputs\first_place_train_fold0.csv --cache-dir C:\ASL\islr_feature_cache_fp16 --batch-size 32
```

If you built only a partial cache and want the check to report the usable cached subset, add:

```powershell
python scripts\check_cached_dataloader.py --csv outputs\first_place_train_fold0.csv --cache-dir C:\ASL\islr_feature_cache_fp16 --batch-size 32 --filter-missing-cache
```

Use cached features for the tiny baseline:

```powershell
python src\train_baseline.py --config configs\tiny_baseline_cached.json
```

For cached training, make sure cache files exist for the rows that training and validation will read. The 1000-row cache command above is intended as a small cache test. `configs/tiny_baseline_cached.json` uses `filter_missing_cache=true`, so it will train only on rows that already have `.npy` files. If validation has no cached rows, build cache for the valid CSV too:

```powershell
python scripts\build_feature_cache.py --csv outputs\first_place_valid_fold0.csv --cache-dir C:\ASL\islr_feature_cache_fp16 --max-samples 640
```

## Baseline Evaluation

Current comparable baseline:

- Tiny cached baseline
- fold0
- 5 epochs
- best valid accuracy around 0.426

Run full validation split evaluation manually:

```powershell
python src\evaluate.py --config configs\tiny_baseline_cached.json --checkpoint outputs\baseline_cached_tiny_fold0_best.pt --split valid
```

Plot training curves manually:

```powershell
python scripts\plot_training_curves.py --csv outputs\baseline_cached_metrics.csv
```

`evaluate.py` does not train; it only loads a checkpoint and evaluates it. `plot_training_curves.py` only reads the metrics CSV and writes PNG figures. If `outputs\baseline_cached_metrics.csv` does not exist yet, rerun `train_baseline.py` manually once to generate it.

## Small Baseline

Small baseline is the next upgrade after the Tiny cached baseline. It stays within a locally trainable size while moving closer to the first-place 1DCNN + Transformer structure:

- More Conv1D blocks
- Two Conv + Transformer stages
- Larger `d_model`
- Still no AWP, SWA, Snapshot, ensemble, or TFLite export

Tiny baseline reference result:

- top1 around 0.426
- top5 around 0.675

Small baseline goal:

- Check whether the larger backbone improves valid top1 / top5 over Tiny.

The first Small run used the `small_cached` output prefix. The current Small config now points to the v2 training recipe below.

### Small Baseline v2

Small baseline v2 keeps `SmallISLRModel` unchanged and only adjusts training strategy:

- `label_smoothing = 0.1`
- `CosineAnnealingLR`
- best checkpoint selected by `valid_acc`
- new output prefix: `small_cosine_ls`

The goal is to reduce overfitting in the Small model and move closer to the first-place cosine decay plus regularized training recipe, without adding AWP, SWA, Snapshot, ensemble, or TFLite export.

Train manually:

```powershell
python src\train_baseline.py --config configs\small_baseline_cached.json
```

Plot curves manually:

```powershell
python scripts\plot_training_curves.py --csv outputs\small_cosine_ls_metrics.csv
```

Evaluate manually:

```powershell
python src\evaluate.py --config configs\small_baseline_cached.json --checkpoint outputs\small_cosine_ls_fold0_best.pt --split valid
```

### Small Baseline v2 max_len=128 + augmentation

This experiment builds on Small v2 with `max_len=128` cached features and adds lightweight cached feature augmentation to reduce the train/valid gap. The previous Small v2 `max_len=128` reference result was best `valid_acc` around `0.5368`.

The augmentation is applied only during training. Validation and evaluation use the cached features unchanged.

Train manually:

```powershell
python src\train_baseline.py --config configs\small_v2_maxlen128_aug_cached.json
```

Plot curves manually:

```powershell
python scripts\plot_training_curves.py --csv outputs\small_v2_len128_aug_metrics.csv
```

Evaluate manually:

```powershell
python src\evaluate.py --config configs\small_v2_maxlen128_aug_cached.json --checkpoint outputs\small_v2_len128_aug_fold0_best.pt --split valid
```

You can override the dataset location:

```powershell
python scripts/check_dataset.py --data-root C:\ASL\asl-signs
python scripts/inspect_sample.py --data-root C:\ASL\asl-signs --index 0
python src/preprocess.py --data-root C:\ASL\asl-signs --index 0
python src/split.py --data-root C:\ASL\asl-signs --n-splits 5
python scripts/check_dataloader.py --data-root C:\ASL\asl-signs --fold 0 --batch-size 8
python scripts/check_first_place_preprocess.py --data-root C:\ASL\asl-signs --fold 0 --batch-size 8
python -m unittest tests.test_model_tiny -v
python src/train_smoke.py --data-root C:\ASL\asl-signs --fold 0 --batch-size 16 --max-train-batches 20 --max-valid-batches 5 --epochs 1
python src/train_baseline.py --config configs/tiny_baseline.json
```

`check_first_place_preprocess.py` checks both first-place preprocessing center modes by default:

- `notebook_strict`: uses landmark `[17]` as the center reference, matching `ISLR_1st_place_Hoyeol_Sohn.ipynb`
- `nose_mean`: uses the mean of `NOSE = [1, 2, 98, 327]`

## Outputs

- `outputs/dataset_summary.txt`
- `outputs/sample_inspection.txt`
- `outputs/env_check.txt`
- `outputs/train_with_folds.csv`
- `outputs/split_summary.txt`
- `outputs/dataloader_check.txt`
- `outputs/first_place_preprocess_check.txt`
- `outputs/smoke_train_log.txt`
- `outputs/smoke_model.pt`
- `outputs/baseline_train_log.txt`
- `outputs/baseline_tiny_fold0_best.pt`
- `outputs/baseline_tiny_fold0_last.pt`
- `outputs/baseline_cached_metrics.csv`
- `outputs/eval_tiny_baseline_valid.json`
- `outputs/eval_tiny_baseline_per_class.csv`
- `outputs/figures/tiny_baseline_loss_curve.png`
- `outputs/figures/tiny_baseline_acc_curve.png`
- `outputs/small_cached_train_log.txt`
- `outputs/small_cached_metrics.csv`
- `outputs/small_cached_fold0_best.pt`
- `outputs/small_cached_fold0_last.pt`
- `outputs/small_cosine_ls_train_log.txt`
- `outputs/small_cosine_ls_metrics.csv`
- `outputs/small_cosine_ls_fold0_best.pt`
- `outputs/small_cosine_ls_fold0_last.pt`
- `outputs/small_v2_len128_aug_train_log.txt`
- `outputs/small_v2_len128_aug_metrics.csv`
- `outputs/small_v2_len128_aug_fold0_best.pt`
- `outputs/small_v2_len128_aug_fold0_last.pt`
- `outputs/cache_metadata.csv`

## Scope

Included:

- Dataset structure checks
- `train.csv` and label map inspection
- Single parquet sample inspection
- Minimal landmark tensor preprocessing
- Participant GroupKFold split
- PyTorch Dataset and DataLoader sanity checks
- First-place-style preprocessing check with `[64, 708]` features
- Tiny 1DCNN + Transformer smoke training to verify forward, loss, backward, GPU use, and checkpoint saving
- Config-driven fold0 tiny baseline training with best-checkpoint saving
- Small trial-training limits in `configs/tiny_baseline.json` for quick baseline checks
- Optional `.npy` first-place feature cache and cached Dataset/DataLoader
- Baseline evaluation JSON, per-class CSV, and training curve plotting utilities
- Small 1DCNN + Transformer cached baseline config and model
- Small baseline v2 training strategy with label smoothing, cosine LR decay, and best-by-accuracy checkpointing
- Small v2 max_len=128 cached feature augmentation config

Not included yet:

- Multi-fold training
- Ensemble
- AWP, SWA, Snapshot
- TFLite export
- Full competition notebook reproduction
- Raspberry Pi deployment
- UI
