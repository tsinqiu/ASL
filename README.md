# ASL Isolated Sign Raspberry Pi Deployment

This branch is a clean deployment branch for running the Kaggle Google Isolated Sign Language Recognition model on Raspberry Pi / Linux ARM.

The model recognizes ASL isolated signs. Chinese text, when shown, is only a display-time meaning for the English ASL label. It is not Chinese Sign Language recognition, and it is not continuous sign language translation.

## What Is Included

- `raspi_deploy/`: Raspberry Pi runtime package
- `scripts/export_onnx.py`: PC-side PyTorch checkpoint to ONNX export
- `configs/small_baseline_cached.json`: model architecture and `max_frames` config used by the export script
- `src/model_small.py`: Small model definition needed to load the checkpoint for export
- `src/model_tiny.py`: Tiny model definition retained because the exporter supports both `tiny` and `small`
- `docs/raspi_deployment_report_notes.md`: report notes for the deployment section

Training code, dataset inspection scripts, parquet preprocessing utilities, cache builders, tests, and notebooks are intentionally removed from this branch.

## PC-Side ONNX Export

Run this manually on the PC before copying the deployment package to Raspberry Pi:

```powershell
python scripts\export_onnx.py --config configs\small_baseline_cached.json --checkpoint outputs\small_cosine_ls_fold0_best.pt --output raspi_deploy\model.onnx
```

If the best checkpoint changes, replace the `--checkpoint` path. The config and checkpoint must match the same model architecture and `max_frames`.

The exporter prints:

- checkpoint path
- output ONNX path
- max length
- input dimension
- number of classes

## Raspberry Pi Runtime

Follow:

- `raspi_deploy/README_RASPI.md`

Typical copy command after `raspi_deploy/model.onnx` exists:

```bash
scp -r raspi_deploy pi@<raspi_ip>:/home/pi/asl_demo
```

Typical Raspberry Pi run command for the CSI camera with Picamera2:

```bash
cd /home/pi/asl_demo
sudo apt update
sudo apt install -y python3-picamera2
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-raspi.txt
python realtime_asl_raspi.py --model model.onnx --camera-backend picamera2 --camera 0 --max-len 64
```

The deployment config defaults to the observed CSI camera corrections: `rotate_180=true` and `swap_r_g=true`.

Controls:

- `r`: start recording one isolated sign action
- `s`: stop recording and recognize
- `q`: quit

## Runtime Flow

```text
Camera image
-> MediaPipe Holistic landmarks
-> restore Kaggle [T, 543, 3] landmark order
-> runtime first-place-style preprocessing to [max_len, 708]
-> ONNX Runtime inference
-> CLI top1/top5 ASL label output with optional Chinese meaning
```

## Boundaries

- No retraining in this branch.
- No training code in this branch.
- No recommendation vocabulary.
- No complex UI.
- No web service.
- No continuous sentence recognition.
- Chinese meanings are label explanations only and are not used for training.

## Report Notes

See:

- `docs/raspi_deployment_report_notes.md`
