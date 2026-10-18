# Power-line point classification in airborne LiDAR

A neural network that finds **power-line conductor points** (ASPRS class 14, *Wire – Conductor*) in airborne LiDAR
point clouds (LAS). Written in 2024; kept as an early example of applying deep learning to 3D geospatial data.

## Method

- **Input features per point:** height `z`, intensity, return number, number of returns, and RGB.
- **Target:** binary label, conductor (class 14) or not, taken from the existing classification.
- **Model:** fully connected network (128 → 64 → dropout → sigmoid) in TensorFlow/Keras with Adam and binary cross-entropy.
- **Imbalance handling:** early stopping on PR-AUC, which suits a rare class.
- **Data split:** train/validation/test split with MinMax scaling.
- **Inference:** the model scores every point, applies a 0.7 probability threshold, and exports the coordinates of detected points.

| File | Purpose |
|---|---|
| [`Model.py`](Model.py) | load LAS files, build features, train and save the model |
| [`Model_prediction.py`](Model_prediction.py) | load a trained model, classify a LAS file, export detected points |

## Run

```bash
pip install -r requirements.txt
python Model.py              # expects TRAIN_DATA.las and vall_small.las in the working directory
python Model_prediction.py   # expects a saved model in ./MODEL
```

The LiDAR data are not included in the repository.

## Known limitations

- Features are per point only. There are no neighbourhood or geometric descriptors (linearity, local height above ground),
  which are usually what separates wires from vegetation.
- The prediction script refits the scaler on the new file instead of reusing the training scaler.
- The file used for prediction was also part of the training data, so there is no independent test result.

**Status:** 2024, not maintained.

## License

[MIT](LICENSE)
