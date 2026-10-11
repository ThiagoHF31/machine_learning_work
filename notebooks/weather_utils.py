"""Helpers for weather_prediction_macau.ipynb: data, quality, features, models and evaluation."""
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.base import is_classifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier,
                              GradientBoostingRegressor, GradientBoostingClassifier,
                              HistGradientBoostingRegressor, HistGradientBoostingClassifier,
                              VotingRegressor, VotingClassifier)
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, accuracy_score,
                             balanced_accuracy_score, f1_score, precision_recall_fscore_support)

CLASSES = ["Clear", "Partly Cloudy", "Cloudy", "Rain"]
NUMERIC_COLS = ["tempmax", "tempmin", "temp", "feelslikemax", "feelslikemin", "dew", "humidity",
                "precip", "precipcover", "windgust", "windspeed", "winddir",
                "sealevelpressure", "cloudcover", "solarenergy"]

# Colorblind-safe palette; one fixed color per class in every chart
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984"
CLASS_COLORS = dict(zip(CLASSES, [C_BLUE, C_YELLOW, C_AQUA, C_ORANGE]))


def set_plot_style():
    """Light grid, thin lines, no top/right spines."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 100, "axes.spines.top": False, "axes.spines.right": False,
                         "lines.linewidth": 1.5, "grid.alpha": 0.4})


def make_dirs(*paths):
    """Create the folders if missing and return them."""
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
    return paths


# ---------------------------------------------------------------- data
# Open-Meteo daily variable -> Visual Crossing column
OPEN_METEO_TO_VC = {
    "time": "datetime",
    "temperature_2m_max": "tempmax", "temperature_2m_min": "tempmin", "temperature_2m_mean": "temp",
    "apparent_temperature_max": "feelslikemax", "apparent_temperature_min": "feelslikemin",
    "dew_point_2m_mean": "dew", "relative_humidity_2m_mean": "humidity",
    "precipitation_sum": "precip", "precipitation_hours": "precipcover",
    "wind_gusts_10m_max": "windgust", "wind_speed_10m_max": "windspeed",
    "wind_direction_10m_dominant": "winddir", "pressure_msl_mean": "sealevelpressure",
    "cloud_cover_mean": "cloudcover", "shortwave_radiation_sum": "solarenergy",
}


def check_coverage(df, name, date_col="datetime", min_days=3000):
    """Print days, period, gaps and future dates; True if usable for training."""
    dates = pd.to_datetime(df[date_col])
    gaps = len(pd.date_range(dates.min(), dates.max()).difference(dates))
    future = dates.max() > pd.Timestamp.today().normalize()
    ok = len(dates) >= min_days and not gaps and not future
    print(f"{name}: {len(dates)} days | {dates.min().date()} -> {dates.max().date()} | missing dates: {gaps} | "
          f"future dates (forecast): {'yes' if future else 'no'} | usable for training: {'YES' if ok else f'NO (needs >= {min_days} days)'}")
    return ok


def download_open_meteo(lat, lon, start, end, timezone):
    """Daily ERA5 history from the Open-Meteo Archive API (original column names)."""
    params = dict(latitude=lat, longitude=lon, start_date=start, end_date=end, timezone=timezone,
                  daily=",".join(list(OPEN_METEO_TO_VC)[1:]))
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=180)
    r.raise_for_status()
    return pd.DataFrame(r.json()["daily"])


def make_vc_style_conditions(precip, cloudcover):
    """Visual Crossing-style label: rain >= 1 mm (WMO rain day); clouds >= 70% overcast, >= 30% partly."""
    sky = np.select([cloudcover >= 70, cloudcover >= 30], ["Overcast", "Partially cloudy"], "Clear")
    return np.where(precip >= 1.0, "Rain, " + sky, sky)


def to_visual_crossing_schema(om, location="Macau"):
    """Open-Meteo raw -> Visual Crossing format (column names, % of day with rain, 'conditions')."""
    df = om.rename(columns=OPEN_METEO_TO_VC)
    df["precipcover"] = (df["precipcover"] / 24 * 100).round(2)
    df["conditions"] = make_vc_style_conditions(df["precip"], df["cloudcover"])
    df.insert(0, "name", location)
    return df[["name", "datetime"] + NUMERIC_COLS + ["conditions"]]


def to_metric(df):
    """Visual Crossing US units (°F, in, mph) -> °C, mm, km/h; no-op if already metric."""
    if df["tempmax"].median() <= 45:                     # no place has a median Tmax above 45 °C
        return df
    df = df.copy()
    temps = ["tempmax", "tempmin", "temp", "feelslikemax", "feelslikemin", "feelslike", "dew"]
    df[temps] = ((df[temps] - 32) * 5 / 9).round(1)
    df["precip"] = (df["precip"] * 25.4).round(2)
    df[["windgust", "windspeed"]] = (df[["windgust", "windspeed"]] * 1.609344).round(1)
    return df


# ---------------------------------------------------------------- quality
NON_NEGATIVE = ["precip", "precipcover", "windgust", "windspeed", "winddir", "sealevelpressure",
                "cloudcover", "solarenergy", "humidity"]
PERCENT = ["humidity", "precipcover", "cloudcover"]


def quality_report(df):
    """Count data problems using physical rules per variable type (no per-column limits)."""
    num, dates = df[NUMERIC_COLS], pd.to_datetime(df["datetime"])
    return pd.Series({
        "duplicated dates": dates.duplicated().sum(),
        "missing dates": len(pd.date_range(dates.min(), dates.max()).difference(dates)),
        "missing values (NaN)": num.isna().sum().sum(),
        "negative where must be >= 0": (num[NON_NEGATIVE] < 0).sum().sum(),
        "percent above 100%": (num[PERCENT] > 100).sum().sum(),
        "wind direction above 360°": (num["winddir"] > 360).sum(),
        "Tmin above Tmax": (num["tempmin"] > num["tempmax"]).sum(),
        "Tmean outside [Tmin, Tmax]": (~num["temp"].between(num["tempmin"], num["tempmax"])).sum(),
    }, name="count").astype(int)


def iqr_outliers(df, cols, k=3.0):
    """Values beyond k*IQR from the quartiles (k=3: extreme outliers)."""
    q1, q3 = df[cols].quantile(0.25), df[cols].quantile(0.75)
    lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
    return pd.DataFrame({"lower": lo.round(1), "upper": hi.round(1),
                         "n_outliers": ((df[cols] < lo) | (df[cols] > hi)).sum()})


def to_class(labels):
    """Raw label -> 4 classes; first match wins (any precipitation -> Rain)."""
    s = labels.str.lower()
    rules = [s.str.contains("rain|snow|freezing|drizzle|thunder|hail|shower"), s.str.contains("overcast"),
             s.str.contains("partially cloudy|partly cloudy"), s.str.contains("clear")]
    return pd.Series(np.select(rules, ["Rain", "Cloudy", "Partly Cloudy", "Clear"], None), index=labels.index)


# ---------------------------------------------------------------- features
LAG_VARS = ["tempmax", "tempmin", "temp", "dew", "humidity", "precip",
            "sealevelpressure", "cloudcover", "windspeed", "solarenergy"]
ROLL_VARS = ["tempmax", "tempmin", "temp", "humidity", "precip", "sealevelpressure", "cloudcover"]


def build_features(d):
    """Features from days t, t-1, ... and targets for day t+1 (no leakage). Returns (table, feature names)."""
    p, doy, month = d["sealevelpressure"], d.index.dayofyear, d.index.month
    f = {c: d[c] for c in NUMERIC_COLS if c != "winddir"}                                   # today
    f |= {f"{c}_lag{k}": d[c].shift(k) for c in LAG_VARS for k in (1, 2)}                    # lags
    f |= {"tempmax_diff1": d["tempmax"].diff(), "tempmin_diff1": d["tempmin"].diff(),         # trends
          "pressure_diff1": p.diff(), "pressure_diff3": p.diff(3), "humidity_diff1": d["humidity"].diff()}
    f |= {f"{c}_roll{w}": d[c].rolling(w).mean() for c in ROLL_VARS for w in (3, 7, 14)}      # rolling means
    f |= {"temp_std7": d["temp"].rolling(7).std(), "pressure_std7": p.rolling(7).std(),
          "rain_days7": d["cond_class"].eq("Rain").astype(int).rolling(7).sum(),
          "temp_range": d["tempmax"] - d["tempmin"], "dewpoint_dep": d["temp"] - d["dew"],     # physics
          "temp_anomaly30": d["temp"] - d["temp"].rolling(30).mean(),
          "doy_sin": np.sin(2 * np.pi * doy / 365.25), "doy_cos": np.cos(2 * np.pi * doy / 365.25),  # cycles
          "month_sin": np.sin(2 * np.pi * month / 12), "month_cos": np.cos(2 * np.pi * month / 12),
          "winddir_sin": np.sin(np.deg2rad(d["winddir"])), "winddir_cos": np.cos(np.deg2rad(d["winddir"]))}
    f |= {f"today_{c.replace(' ', '_')}": d["cond_class"].eq(c).astype(int) for c in CLASSES}  # one-hot
    feat = pd.DataFrame(f, index=d.index)
    names = list(feat.columns)
    feat = feat.assign(target_tmax=d["tempmax"].shift(-1), target_tmin=d["tempmin"].shift(-1),
                       target_cond=d["cond_class"].shift(-1), target_date=d.index + pd.Timedelta(days=1))
    return feat.dropna(), names


# ---------------------------------------------------------------- models
def scaled(model):
    """Scale-sensitive model behind a StandardScaler (fitted on training data only)."""
    return make_pipeline(StandardScaler(), model)


def make_regressors(seed=42):
    """Fresh regressors with initial hyperparameters."""
    trees = dict(n_estimators=300, min_samples_leaf=2, max_features=0.5, n_jobs=-1, random_state=seed)
    return {
        "LinearRegression": scaled(LinearRegression()),
        "Ridge": scaled(Ridge(alpha=1.0)),
        "Lasso": scaled(Lasso(alpha=0.01, max_iter=10000)),
        "KNN": scaled(KNeighborsRegressor(n_neighbors=20, weights="distance")),
        "SVR": scaled(SVR(C=10, epsilon=0.1)),
        "RandomForest": RandomForestRegressor(**trees),
        "ExtraTrees": ExtraTreesRegressor(**trees),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=seed),
        "HistGradBoost": HistGradientBoostingRegressor(learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
                                                       min_samples_leaf=20, random_state=seed),
    }


def make_classifiers(seed=42):
    """Fresh classifiers; class_weight balances classes where supported."""
    trees = dict(n_estimators=400, min_samples_leaf=3, max_features="sqrt", class_weight="balanced_subsample",
                 n_jobs=-1, random_state=seed)
    return {
        "LogisticReg": scaled(LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=seed)),
        "KNN": scaled(KNeighborsClassifier(n_neighbors=30, weights="distance")),
        "SVC": scaled(SVC(C=1.0, class_weight="balanced", probability=True, random_state=seed)),
        "NaiveBayes": scaled(GaussianNB()),
        "RandomForest": RandomForestClassifier(**trees),
        "ExtraTrees": ExtraTreesClassifier(**trees),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=seed),
        "HistGradBoost": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=31,
                                                        min_samples_leaf=20, class_weight="balanced", random_state=seed),
    }


# Tuning grids, one per model (any model can win validation).
# Pipeline steps are named after the lowercased class, e.g. "ridge__alpha".
HGB_GRID = {"learning_rate": [0.02, 0.05, 0.1], "max_leaf_nodes": [15, 31], "min_samples_leaf": [20, 50]}
GB_GRID = {"learning_rate": [0.05, 0.1], "max_depth": [2, 3]}
REG_TREES = {"n_estimators": [300], "max_depth": [None, 12, 20], "min_samples_leaf": [1, 2, 5], "max_features": [0.3, 0.5, 0.7]}
CLF_TREES = {"n_estimators": [400], "max_depth": [None, 10, 20], "min_samples_leaf": [1, 4, 10], "max_features": [0.1, 0.2, "sqrt"]}
REG_PARAM_GRIDS = {
    "LinearRegression": {"linearregression__fit_intercept": [True]},
    "Ridge": {"ridge__alpha": [0.01, 0.1, 1, 10, 30, 100, 300, 1000]},
    "Lasso": {"lasso__alpha": [0.001, 0.003, 0.01, 0.03, 0.1]},
    "KNN": {"kneighborsregressor__n_neighbors": [5, 10, 20, 40], "kneighborsregressor__weights": ["uniform", "distance"]},
    "SVR": {"svr__C": [1, 10, 30], "svr__epsilon": [0.1, 0.5]},
    "RandomForest": REG_TREES,
    "ExtraTrees": REG_TREES,
    "GradientBoosting": {"n_estimators": [200, 400], **GB_GRID},
    "HistGradBoost": {"max_iter": [300, 600], **HGB_GRID},
}
CLF_PARAM_GRIDS = {
    "LogisticReg": {"logisticregression__C": [0.001, 0.01, 0.1, 1, 10]},
    "KNN": {"kneighborsclassifier__n_neighbors": [10, 30, 50, 100], "kneighborsclassifier__weights": ["uniform", "distance"]},
    "SVC": {"svc__C": [0.3, 1, 3]},
    "NaiveBayes": {"gaussiannb__var_smoothing": [1e-9, 1e-7, 1e-5]},
    "RandomForest": CLF_TREES,
    "ExtraTrees": CLF_TREES,
    "GradientBoosting": {"n_estimators": [100, 200], **GB_GRID},
    "HistGradBoost": {"max_iter": [200, 400], **HGB_GRID},
}


def tune_model(model, grid, X, y, scoring, n_folds=5):
    """Grid search with 5 time-ordered folds (TimeSeriesSplit, 1-day gap); shuffled KFold would leak the future."""
    return GridSearchCV(model, grid, scoring=scoring, n_jobs=-1, cv=TimeSeriesSplit(n_folds, gap=1)).fit(X, y)


def fmt_params(params):
    """Best params with rounded floats, for printing."""
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in params.items()}


def ensemble(members):
    """Average of the members: mean prediction (regression) or mean probability / soft voting (classification)."""
    est = list(members.items())
    return VotingClassifier(est, voting="soft") if is_classifier(est[0][1]) else VotingRegressor(est)


def climatology_predict(train_df, target_df, var):
    """Training mean per day of year, smoothed with a circular 15-day window."""
    clim = train_df.groupby(train_df["target_date"].dt.dayofyear)[var].mean().reindex(range(1, 367)).interpolate()
    clim = pd.concat([clim.iloc[-7:], clim, clim.iloc[:7]]).rolling(15, center=True).mean().iloc[7:-7]
    return clim.loc[target_df["target_date"].dt.dayofyear].values


def persistence_class(part):
    """Tomorrow's class = today's class (from the one-hot columns)."""
    onehot = part[[f"today_{c.replace(' ', '_')}" for c in CLASSES]]
    return np.array(CLASSES)[onehot.values.argmax(axis=1)]


# ---------------------------------------------------------------- evaluation
def regression_metrics(y_true, y_pred):
    """MAE, RMSE, R² and Pearson r."""
    return {"MAE": mean_absolute_error(y_true, y_pred), "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2": r2_score(y_true, y_pred), "Pearson_r": pearsonr(y_true, y_pred)[0]}


def f1_macro(y_true, y_pred):
    """F1 averaged over the 4 classes with equal weight."""
    return f1_score(y_true, y_pred, average="macro", labels=CLASSES, zero_division=0)


def classification_metrics(y_true, y_pred):
    """F1-macro, accuracy, balanced accuracy and weighted F1."""
    return {"F1_macro": f1_macro(y_true, y_pred), "Accuracy": accuracy_score(y_true, y_pred),
            "Balanced_acc": balanced_accuracy_score(y_true, y_pred),
            "F1_weighted": f1_score(y_true, y_pred, average="weighted", labels=CLASSES, zero_division=0)}


def rain_metrics(y_true, y_pred):
    """Rain vs no rain: accuracy, precision, recall and F1 of Rain."""
    t, p = np.asarray(y_true) == "Rain", np.asarray(y_pred) == "Rain"
    prec, rec, f1, _ = precision_recall_fscore_support(t, p, average="binary", zero_division=0)
    return {"Accuracy": accuracy_score(t, p), "Precision_rain": prec, "Recall_rain": rec, "F1_rain": f1}


def block_bootstrap_diff(y_true, pred_a, pred_b, metric, block=7, n_boot=2000, seed=42):
    """95% CI of metric(B) - metric(A) with a 7-day moving block bootstrap; excludes 0 = significant."""
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    n = len(y_true)
    starts = np.random.default_rng(seed).integers(0, n - block + 1, (n_boot, -(-n // block)))
    idx = (starts[..., None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    diffs = [metric(y_true[i], pred_b[i]) - metric(y_true[i], pred_a[i]) for i in idx]
    return np.percentile(diffs, [2.5, 97.5])


FEATURE_GROUPS = [("_lag", "Lags (t-1, t-2)"), ("_roll", "Rolling stats"), ("_std7", "Rolling stats"),
                  ("rain_days7", "Rolling stats"), ("_diff", "Differences"), ("doy_", "Seasonality"),
                  ("month_", "Seasonality"), ("today_", "Today's class"), ("temp_range", "Physical"),
                  ("dewpoint_dep", "Physical"), ("temp_anomaly30", "Physical")]


def feature_group(name):
    """Feature family, to sum importance by group."""
    return next((g for key, g in FEATURE_GROUPS if key in name), "Today's values")
