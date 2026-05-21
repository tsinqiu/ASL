# Raspberry Pi ASL Isolated Sign Deployment

This directory is a clean Raspberry Pi / Linux ARM deployment package for the Kaggle Google Isolated Sign Language Recognition model.

The system recognizes ASL isolated signs. Chinese text is only an English ASL label meaning shown for readability; it is not Chinese sign language recognition and it is not continuous sign language translation.

## Directory Contents

- `realtime_asl_raspi.py`: camera recording demo with ONNX Runtime inference
- `preprocess_runtime.py`: lightweight first-place-style runtime preprocessing, no pandas and no parquet
- `label_utils.py`: label and Chinese meaning loading helpers
- `labels.json`: class index to English ASL sign mapping
- `asl_label_zh_map.json`: optional English ASL sign to Chinese meaning mapping
- `config.json`: default runtime config
- `requirements-raspi.txt`: Python dependencies for Raspberry Pi
- `model.onnx`: exported model file, generated on the PC and copied here

## 1. Export ONNX On The PC

Run this from the project root on the PC:

```powershell
python scripts\export_onnx.py --config configs\small_baseline_cached.json --checkpoint outputs\small_cosine_ls_fold0_best.pt --output raspi_deploy\model.onnx
```

If your best checkpoint is different, replace the `--checkpoint` path. The checkpoint and config must match the same model architecture and `max_frames`.

## 2. Prepare Labels

`labels.json` is already included in this deployment directory. It is the reverse mapping of Kaggle `sign_to_prediction_index_map.json`:

```json
{
  "0": "TV",
  "1": "after",
  "2": "airplane"
}
```

The ONNX model outputs class indices, and `labels.json` maps those indices back to English ASL labels.

## 3. Copy To Raspberry Pi

After `model.onnx` exists, copy the deployment directory to the Raspberry Pi:

```bash
scp -r raspi_deploy pi@<raspi_ip>:/home/pi/asl_demo
```

## 4. Install Dependencies On Raspberry Pi

On the Raspberry Pi:

```bash
cd /home/pi/asl_demo
sudo apt update
sudo apt install -y python3-picamera2
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-raspi.txt
```

`python3-picamera2` is installed through Raspberry Pi OS packages rather than pip. The `--system-site-packages` flag lets the virtual environment see the system Picamera2/libcamera installation.

`onnxruntime` may have Raspberry Pi architecture wheel issues. First try:

```bash
pip install onnxruntime
```

If it fails, install an onnxruntime wheel matching your OS and CPU architecture. If ONNX Runtime remains unavailable, use the PC PyTorch inference demo as a fallback.

## 5. Run

```bash
python realtime_asl_raspi.py --model model.onnx --camera-backend picamera2 --camera 0 --max-len 64
```

The default `config.json` is set for the CSI camera path:

```json
{
  "camera_backend": "picamera2",
  "rotate_180": true,
  "swap_r_g": true
}
```

These defaults match the observed hardware setup where the CSI camera image needs a 180 degree rotation and red/green channel swap before landmark extraction. For a USB camera or PC-style OpenCV camera, use:

```bash
python realtime_asl_raspi.py --model model.onnx --camera-backend opencv --camera 0 --max-len 64 --no-rotate-180 --no-swap-r-g
```

Controls:

- `r`: start recording one isolated sign action
- `s`: stop recording and recognize
- `q`: quit

Output example:

```text
Top1: milk / 牛奶 confidence=0.7200
Top5:
1. milk / 牛奶 confidence=0.7200
2. drink / 喝 confidence=0.1100
3. water / water confidence=0.0500
4. go / 去 confidence=0.0300
5. yes / 是 confidence=0.0200
```

## 6. MediaPipe Notes

The script first tries `mp.solutions.holistic`. Some newer MediaPipe Python packages expose only the Tasks API. In that case, provide a HolisticLandmarker `.task` file:

```bash
python realtime_asl_raspi.py --model model.onnx --camera 0 --max-len 64 --mediapipe-task /home/pi/asl_demo/holistic_landmarker.task
```

## 7. Project Boundary

- The model recognizes ASL isolated signs from the Kaggle dataset.
- Chinese meanings are only display-time label explanations.
- Chinese meanings are not used for training.
- This is not a Chinese Sign Language dataset/model.
- This is not continuous sign language translation.
- The demo uses recording-based isolated sign classification, not continuous sliding-window sentence recognition.

## 8. Performance Notes

MediaPipe Holistic can be slow on Raspberry Pi. This deployment intentionally uses recording-based recognition instead of continuous real-time sliding windows. If frame processing is too slow, lower camera resolution outside this script or reduce `record_fps_limit` in `config.json`.
