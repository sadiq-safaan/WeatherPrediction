# Rainfall Forecasting

Predicts rain rate in mm/hr and rain probability (>= 1 mm/hr) at 1/3/5 minute horizons using GOES and HRRR data.

## Files

`train3.py`: Trains the DoubleEncoder MLP models and saves weights and scalers
`predict.py`: Runs inference on a parquet file
`preprocess3.py`: Data processing pipeline shared by train and predict


## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
python train3.py
```

Expects `train_rows.parquet` and `val_rows.parquet` in the working directory. Saves the following:

- `model_1.pt`, `model_3.pt`, `model_5.pt` — artefacts for t+1, t+3, t+5 horizons
- `sg.joblib`, `sh.joblib` — scalers (Robust Scaler) for rain and cloud features

## Inference

```bash
python predict.py --input /path/to/heldout.parquet --output /path/to/predictions.parquet
```

Requires the trained model weights and scalers in the working directory. 
Output is a parquet file with columns: 

`site_id`, `timestamp`, `horizon_minute`, `rain_probability`, `rain_rate_mm_hr` 

sorted by `site_id, timestamp, horizon_minute`.
