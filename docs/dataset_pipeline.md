# Dataset and DataLoader Pipeline

This stage builds the data infrastructure for Kaggle "Google - Isolated Sign Language Recognition" without training a model.

## Source Files

- `train.csv` is the sample index table. Each row contains a parquet `path`, `participant_id`, `sequence_id`, and `sign`.
- `sign_to_prediction_index_map.json` is the label map from sign strings to integer class ids.
- Each parquet file is one isolated sign sample represented as a long table of MediaPipe landmark coordinates, not as raw video.

## Dataset

`ISLRDataset` converts:

`train.csv + parquet -> x, mask, y`

- `x`: `torch.FloatTensor` with shape `[max_frames, feature_dim]`
- `mask`: `torch.BoolTensor` with shape `[max_frames]`
- `y`: integer class label from `sign_to_prediction_index_map.json`

The default input keeps `left_hand + right_hand + pose` and excludes `face` for now. With MediaPipe landmark counts, the default feature dimension is:

`(21 + 21 + 33) * 3 = 225`

The default `max_frames` is `64`. Real frames are marked `True` in `mask`; padded frames are marked `False`.

## GroupKFold

`src/split.py` uses `GroupKFold` with `participant_id` as the group key. This prevents the same signer from appearing in both train and validation splits, reducing participant leakage.

## DataLoader

`scripts/check_dataloader.py` builds train and validation `DataLoader` instances from a selected fold and checks that samples and batches can be produced.

The DataLoader stage only validates shapes, labels, masks, and NaN handling. It does not train a model.

## First-Place-Style Preprocessing

`src/first_place_preprocess.py` implements the preprocessing pattern from `ISLR_1st_place_Hoyeol_Sohn.ipynb`:

- restore each parquet long table to `[T, 543, 3]`
- select `LIP + LHAND + RHAND + NOSE + REYE + LEYE`
- keep only `x, y`
- concatenate `xy`, first-order difference `dx`, and second-order difference `dx2`
- pad or truncate to `[64, 708]`
- replace NaN values with `0`

The default `center_mode` is `notebook_strict`, which uses landmark `[17]` as the center reference, matching the first-place notebook code. The optional `center_mode="nose_mean"` uses the mean of `NOSE = [1, 2, 98, 327]` as a more explicit nose reference variant.

`scripts/check_first_place_preprocess.py` checks both modes by default and confirms that both produce `[64, 708]` samples and `[batch, 64, 708]` batches.

## Scope

The current task is isolated sign classification. It is not continuous sign language translation and not Chinese Gloss-to-Chinese sentence generation.
