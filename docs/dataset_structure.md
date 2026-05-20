# Kaggle ISLR Dataset Structure

This project uses the Kaggle "Google - Isolated Sign Language Recognition" dataset through a `DATA_ROOT` path, defaulting to `C:\ASL\asl-signs`.

## Files

- `train.csv` is the global index table. Each row points to one isolated sign sample and includes fields such as `path`, `participant_id`, `sequence_id`, and `sign`.
- `sign_to_prediction_index_map.json` maps each sign label string to the integer class id used for classification.
- `train_landmark_files/` contains participant folders. The numeric folder names correspond to `participant_id` values.
- Each `.parquet` file under `train_landmark_files/<participant_id>/` is one isolated sign sample.

## Parquet Contents

The parquet files are not raw videos. They store MediaPipe landmark coordinates in long-table form, typically with columns:

- `frame`
- `row_id`
- `type`
- `landmark_index`
- `x`
- `y`
- `z`

The `type` column identifies landmark groups such as `face`, `pose`, `left_hand`, and `right_hand`.

## Minimal Training Data Flow

The later classification pipeline should follow this shape:

`train.csv -> path -> parquet -> preprocessing -> landmark tensor -> classification label`

The current stage stops at data validation and minimal tensor preprocessing. This is isolated sign classification, not continuous sign language translation, and not Chinese Gloss-to-Chinese sentence generation.
