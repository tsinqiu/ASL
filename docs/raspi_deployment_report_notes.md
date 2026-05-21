# Raspberry Pi Deployment Report Notes

## Deployment Goal

The deployment goal is to migrate the ASL isolated sign recognition model trained on the PC to a Raspberry Pi environment. The Raspberry Pi version should support camera capture, MediaPipe landmark extraction, first-place-style feature preprocessing, ONNX Runtime inference, and recognition result output.

## Technical Flow

```text
Camera image
-> MediaPipe Holistic extracts 543 landmarks
-> Restore Kaggle order [T, 543, 3]
-> Select LIP / HANDS / NOSE / EYES landmarks
-> Build [max_len, 708] features with x,y + dx + dx2
-> ONNX Runtime inference
-> Output ASL label and Chinese meaning
```

## Task Boundary

- The recognition target is ASL isolated sign.
- The Chinese text is only a meaning for the English ASL label.
- The system is not a Chinese Sign Language recognition model.
- The system is not a continuous sign language translation system.
- The deployment does not add recommendation vocabulary, a web service, or a complex UI.

## Why ONNX

The training side uses PyTorch. Raspberry Pi deployment uses ONNX Runtime to avoid installing and maintaining a full PyTorch runtime on Linux ARM. This makes the deployment package smaller and cleaner, while keeping the model export path straightforward.

## Why Recording-Based Recognition

The Kaggle task is isolated sign classification, so a short recorded action is a natural input unit. Raspberry Pi compute is limited, and MediaPipe Holistic can be expensive. Recording-based recognition avoids the heavier cost and ambiguity of continuous real-time sliding-window recognition.

## CSI Camera Notes

The Raspberry Pi deployment uses the CSI camera through Picamera2 by default. In the tested hardware setup, the captured frame requires a 180 degree rotation and a red/green channel swap before MediaPipe landmark extraction. These corrections are configured in `raspi_deploy/config.json` as `rotate_180=true` and `swap_r_g=true`.

## Runtime Files

- `model.onnx`: exported on the PC from the final Small v2 `max_len=128` PyTorch checkpoint
- `realtime_asl_raspi.py`: Raspberry Pi camera demo
- `preprocess_runtime.py`: lightweight NumPy preprocessing
- `labels.json`: model output index to ASL English label
- `asl_label_zh_map.json`: optional ASL English label to Chinese meaning
- `requirements-raspi.txt`: Raspberry Pi Python dependencies
