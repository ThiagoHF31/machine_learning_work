# Next-Day Weather Prediction for Macau – CSAI3121 Group 7

Thiago Honorato Ferreira (P2627421) · Marcelo Oliveira Volpato (P2627394) · João Vitor Konrad (P2627402)

Regression of tomorrow's maximum and minimum temperature and classification of tomorrow's weather condition (Clear, Partly Cloudy, Cloudy, Rain) for Macau, using 12 years of daily ERA5 data (2014–2025).

## Folder structure

```
notebooks/
  weather_prediction_macau.ipynb   main analysis, from data loading to final evaluation
  weather_utils.py                 helper functions (data, quality, features, models, evaluation)
data/                              raw and processed data (see data/README_data.md)
figures/                           figures produced by the notebook
results/                           result tables produced by the notebook
models/                            final fitted models (recreated by the notebook)
```

`project_codes_7.zip` and `project_data_7.zip` share the same root folder. Unzip both into the same directory so that `notebooks/` and `data/` sit side by side; the notebook reads the data from `../data/`.

## How to run

1. Install Python 3.11 or later and the packages in `requirements.txt` (`pip install -r requirements.txt`).
2. Open `notebooks/weather_prediction_macau.ipynb` in Jupyter or VS Code, with the working directory set to `notebooks/`.
3. Run Kernel → Restart & Run All. The full run takes about 3–5 minutes.

The raw Open-Meteo file is included in `data/`. If it is deleted, the notebook downloads it again from the Open-Meteo Archive API (no account needed). All random seeds are fixed, so every run reproduces the tables in `results/`.
