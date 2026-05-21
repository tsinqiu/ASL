# ASL Isolated Sign Recognition

This repository contains an ASL isolated sign recognition project built around
landmark-based sequence classification. It includes data checks, participant
splitting, feature preprocessing, cached feature training, evaluation utilities,
PC webcam inference, ONNX export, and Raspberry Pi deployment preparation.

The model recognizes ASL isolated signs. Chinese text, when shown, is only a
display-time meaning for the English ASL label. It is not Chinese Sign Language
recognition, and it is not continuous sign language translation.

## Data

Default data path:

```powershell
C:\ASL\asl-signs
```

Large source data files are not copied into this repository. Scripts read them
through `--data-root`, which defaults to `C:\ASL\asl-signs`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install pandas numpy pyarrow scikit-learn tqdm
```

Install GPU PyTorch separately with the CUDA wheel index that matches the local
machine:

```powershell
python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```

Environment check:

```powershell
python -c "import sys, torch; print('python:', sys.version); print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('torch cuda:', torch.version.cuda); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

## Common Commands

```powershell
python scripts/check_dataset.py
python scripts/inspect_sample.py --index 0
python src/preprocess.py --index 0
python src/split.py --data-root C:\ASL\asl-signs --n-splits 5
python scripts/check_dataloader.py --data-root C:\ASL\asl-signs --fold 0 --batch-size 8
python -m unittest tests.test_model_tiny -v
python src/train_smoke.py --data-root C:\ASL\asl-signs --fold 0 --batch-size 16 --max-train-batches 20 --max-valid-batches 5 --epochs 1
python src/train_baseline.py --config configs/tiny_baseline.json
```

`configs/tiny_baseline.json` is a small trial-training config. It is useful for
checking the training loop, validation loop, metric logging, and checkpoint
saving before running longer experiments.

## Feature Cache

The runtime feature format is `[max_len, 708]`, based on selected face, hand,
nose, and eye landmarks with `x,y + dx + dx2` features. Cached `.npy` features
make repeated training experiments faster.

Small cache build test:

```powershell
python scripts\build_feature_cache.py --csv outputs\train_fold0.csv --cache-dir C:\ASL\islr_feature_cache_fp16 --max-samples 1000
```

Check cached DataLoader:

```powershell
python scripts\check_cached_dataloader.py --csv outputs\train_fold0.csv --cache-dir C:\ASL\islr_feature_cache_fp16 --batch-size 32
```

If you built only a partial cache and want the check to report the usable cached
subset, add:

```powershell
python scripts\check_cached_dataloader.py --csv outputs\train_fold0.csv --cache-dir C:\ASL\islr_feature_cache_fp16 --batch-size 32 --filter-missing-cache
```

Use cached features for the tiny baseline:

```powershell
python src\train_baseline.py --config configs\tiny_baseline_cached.json
```

## Baseline Evaluation

Current comparable tiny baseline:

- fold0
- cached features
- best valid accuracy around `0.426`

Run full validation split evaluation manually:

```powershell
python src\evaluate.py --config configs\tiny_baseline_cached.json --checkpoint outputs\baseline_cached_tiny_fold0_best.pt --split valid
```

Plot training curves manually:

```powershell
python scripts\plot_training_curves.py --csv outputs\baseline_cached_metrics.csv
```

`evaluate.py` does not train; it only loads a checkpoint and evaluates it.
`plot_training_curves.py` only reads a metrics CSV and writes PNG figures.

## Small Model Experiments

The Small model is the main local training target after the Tiny baseline. It
uses a compact Conv1D + Transformer sequence classifier while remaining practical
for local training.

Tiny baseline reference result:

- top1 around `0.426`
- top5 around `0.675`

Small v2 training strategy:

- `label_smoothing = 0.1`
- `CosineAnnealingLR`
- best checkpoint selected by `valid_acc`
- output prefix: `small_cosine_ls`

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

## Small v2 max_len=128

This experiment keeps the landmark selection and `x,y + dx + dx2` feature
definition unchanged, but increases cached sequence length from `[64, 708]` to
`[128, 708]`.

Use a separate cache directory so the existing `[64, 708]` cache is not
overwritten:

```powershell
C:\ASL\islr_feature_cache_fp16_len128
```

Build the train cache manually:

```powershell
python scripts\build_feature_cache.py --csv outputs\train_fold0.csv --cache-dir C:\ASL\islr_feature_cache_fp16_len128 --dtype float16 --max-len 128 --max-samples 0
```

Build the valid cache manually:

```powershell
python scripts\build_feature_cache.py --csv outputs\valid_fold0.csv --cache-dir C:\ASL\islr_feature_cache_fp16_len128 --dtype float16 --max-len 128 --max-samples 0
```

Check the cached DataLoader manually:

```powershell
python scripts\check_cached_dataloader.py --csv outputs\train_fold0.csv --cache-dir C:\ASL\islr_feature_cache_fp16_len128 --batch-size 32 --max-len 128 --filter-missing-cache
```

Train Small v2 with `max_len=128` manually:

```powershell
python src\train_baseline.py --config configs\small_v2_maxlen128_cached.json
```

Plot curves manually:

```powershell
python scripts\plot_training_curves.py --csv outputs\small_v2_len128_metrics.csv
```

Evaluate manually:

```powershell
python src\evaluate.py --config configs\small_v2_maxlen128_cached.json --checkpoint outputs\small_v2_len128_fold0_best.pt --split valid
```

If GPU memory is not enough, reduce `batch_size` in
`configs\small_v2_maxlen128_cached.json` from `128` to `64`.

## PC Realtime ASL Inference Demo

The PC demo uses a webcam plus MediaPipe landmarks to record one isolated sign
action, convert it into `[1, max_len, 708]` features, run the Small model
checkpoint, and print Top1/Top5 ASL labels. Chinese meanings can be displayed
through `data\asl_label_zh_map.json`.

Install the extra realtime demo dependencies manually:

```powershell
python -m pip install opencv-python mediapipe
```

MediaPipe has two Python API families. Older builds expose
`mp.solutions.holistic` directly. Newer Tasks-only builds expose
`mp.tasks.vision.HolisticLandmarker` and require a separate
`holistic_landmarker.task` model asset. If your installed package reports
`module 'mediapipe' has no attribute 'solutions'`, download a MediaPipe
HolisticLandmarker `.task` model and set this field in
`configs\inference_small_v2.json`:

```json
"mediapipe": {
  "backend": "auto",
  "model_asset_path": "C:\\ASL\\models\\holistic_landmarker.task"
}
```

Configure checkpoint, label map, camera index, and model settings in:

```powershell
configs\inference_small_v2.json
```

Run the PC webcam demo manually:

```powershell
python demo_asl_pc.py --config configs\inference_small_v2.json
```

Controls:

- `r`: start recording one isolated sign action
- `s`: stop recording and recognize
- `q`: quit

Output example:

```text
Top1: wait / wait  confidence=0.7200
Top5:
1. wait / wait  confidence=0.7200
2. later / later  confidence=0.1100
3. thankyou / thankyou  confidence=0.0500
4. stay / stay  confidence=0.0300
5. go / go  confidence=0.0200
```

Optional ONNX export for inference experiments:

```powershell
python scripts\export_onnx.py --config configs\inference_small_v2.json --checkpoint outputs\small_cosine_ls_fold0_best.pt --output outputs\small_cosine_ls.onnx
```

## Outputs

Common generated files include:

- `outputs/dataset_summary.txt`
- `outputs/train_with_folds.csv`
- `outputs/split_summary.txt`
- `outputs/dataloader_check.txt`
- `outputs/smoke_train_log.txt`
- `outputs/baseline_cached_metrics.csv`
- `outputs/small_cosine_ls_metrics.csv`
- `outputs/small_v2_len128_metrics.csv`
- `outputs/small_v2_len128_fold0_best.pt`
- `outputs/small_v2_len128.onnx`
- `outputs/cache_metadata.csv`

## Scope

Included:

- Dataset structure checks
- Label map inspection
- Single sample inspection
- Landmark tensor preprocessing
- Participant GroupKFold split
- PyTorch Dataset and DataLoader checks
- Tiny and Small Conv1D + Transformer models
- Config-driven training and evaluation
- Cached feature Dataset/DataLoader
- `max_len=128` cached feature experiment
- PC realtime ASL isolated sign inference demo
- ONNX export utility

Not included:

- Multi-fold training
- Ensemble
- AWP, SWA, Snapshot
- Complex UI
- Web service
- Continuous sentence recognition
