# Tropical Cyclone Wind Speed Prediction

This project uses historical cyclone observations and environmental weather data
to predict **wind speed 6, 12, and 24 hours ahead**. The central question is whether
a model can improve on simply keeping the current wind speed unchanged.

## 1. Problem statement

Wind intensity changes as a cyclone moves and interacts with its environment.
Knowing its current wind speed is not always enough to determine whether it will
strengthen or weaken. This project combines recent wind history with environmental
variables, compares forecasting methods, and examines where their predictions fail.

This is a historical meteorological study, not an operational warning system
or a financial trading project.

## 2. Inputs and outputs

**Inputs available at forecast origin t:**

- Observed wind speed at t, t-6h, t-12h, and t-18h.
- Position, central pressure, movement speed, and direction where available.
- ERA5 environmental fields at the origin and selected historical timestamps,
  including wind components, humidity, pressure, and related variables.
- Features derived only from these current and past observations.

**Output:** wind speed in meters per second (m/s) at t+6h, t+12h, or t+24h.
A separate model is fitted for each horizon. For example, observations through
12:00 support forecasts for 18:00, 00:00 the following day, and 12:00 the following day.

Realized future weather conditions are not used as inputs. Four observations
spaced six hours apart span **18 hours of history**, from t-18h through t,
rather than a 24-hour historical window.

## 3. Dataset

The existing dataset combines Xweather cyclone records and ERA5 environmental
fields, as documented in the original course project.

| Attribute | Value |
|---|---|
| Original observations | 16,485 |
| Original columns | 56, including metadata and the target |
| Recorded storm IDs | 563 |
| Observation period | March 18, 1980 - November 13, 2025 |
| Target variable | pt_windSpeedMPS |

Provider references: [Xweather](https://www.xweather.com/docs/weather-api/endpoints/tropicalcyclones-archive),
[ERA5 single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels),
and [ERA5 pressure levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels).

The original CSV stays local at data/raw/dataset.csv. A complete original
acquisition script was not present in the supplied project folder. The current
pipeline reuses that CSV and does not claim to reproduce provider acquisition.

## 4. Preprocessing

1. Parse timestamps explicitly as day-month-year and sort by storm and time.
2. Remove equivalent duplicate observations. If observations at the same storm/time
   still disagree, discard those conflicting records rather than selecting or averaging labels.
3. Exclude future-revealing metadata such as lifetime maximum wind, final lifecycle
   statistics, and storm end times.
4. Join histories and labels at **exact clock offsets within the same storm**.
   The next row is not assumed to be six hours later, and future interpolation is not used.
5. Fit missing-value imputation and scaling **only on each training partition**.

Cleaning removed **1,111 equivalent observations** and **six conflicting rows at
three timestamps**, leaving **15,368 observations** before constructing histories
and future labels. See the [data audit](reports/data_audit.json).

## 5. Feature engineering

| Feature group | Purpose |
|---|---|
| Exact-clock lag features | Represent earlier conditions within the same storm |
| Recent wind mean and standard deviation | Summarize intensity and short-term variability |
| Six-hour wind change | Describe recent strengthening or weakening |
| Wind-vector magnitude at pressure levels | Combine the u/v wind components |
| Wind shear between 850 and 200 hPa | Describe differences between wind vectors at two levels |
| Sine/cosine encoding of hour, month, and longitude | Preserve cyclic structure |

Storm ID is a grouping key, not a numerical predictor. The pipeline rebuilds
features from the original CSV rather than trusting the globally preprocessed
feature table from the earlier project.

## 6. Experimental design

| Partition | Period | Role |
|---|---|---|
| Development | Before 2019 | Select model family and hyperparameters |
| Calibration | 2019-2021 | Estimate prediction intervals |
| Holdout test | 2022-2025 | Final evaluation |

Development validation windows cover 2004-2008, 2009-2013, and 2014-2018.
Each fit uses labels from before its validation window. History/label boundary
overlaps are removed, and storm IDs are not shared between training and validation.

Selection minimizes **mean validation MAE across the development windows**.
Models are not reselected using test performance. At each horizon, every model
is evaluated on the same forecast origins and future targets.

## 7. Models

| Model | Main idea |
|---|---|
| Persistence | Future wind equals current observed wind |
| Linear trend | Extrapolate the recent six-hour wind change |
| ElasticNet | Linear regression with regularization |
| Random Forest | Combine multiple regression trees |
| HistGradientBoosting | Fit successive trees to improve predictions |
| PyTorch MLP | Neural network using tabular predictors |
| PyTorch LSTM | Neural network using ordered observation sequences |

LSTM uses a fixed sequence-variable subset; other models may use a broader
feature set. These are comparisons of **model-plus-feature systems**, not
architecture-only experiments with identical inputs.

## 8. Evaluation and results

- **MAE:** average absolute error, directly interpretable in m/s.
- **RMSE:** gives more weight to large errors.
- **R-squared:** explained variation in observed wind speed.
- Additional checks include persistence comparison, strong-wind errors, bias,
  and prediction-interval coverage.

The following results belong to the **corrected common-horizon benchmark**:

| Horizon | Test samples | Persistence MAE | Selected model MAE | MAE reduction |
|---|---:|---:|---:|---:|
| 6 hours | 643 | 3.1042 m/s | 2.4575 m/s | 20.8% |
| 12 hours | 606 | 5.7624 m/s | 4.2161 m/s | 26.8% |
| 24 hours | 538 | 10.1245 m/s | 7.2201 m/s | 28.7% |

HistGradientBoosting was selected at all three horizons using development results.

![Persistence and selected-model MAE](reports/benchmark/figures/benchmark_mae.png)

See [all model results](reports/benchmark/holdout_metrics.csv).
MAE uncertainty is estimated by resampling **whole storms**, rather than treating
successive observations as independent.

## 9. Error analysis and limitations

- Persistence is a strong baseline, especially at short horizons.
- The selected models reduce overall MAE on all three horizons in this historical holdout.
- Removing lag features or ERA5 increases development MAE under the fixed selected
  configurations: [ablation results](reports/benchmark/development_ablations.csv).
- Nominal 90% prediction intervals achieve only **80.9-83.2%** empirical test coverage.
  Calibration does not maintain its nominal coverage in the later period.
- Errors increase at longer horizons, and strong-wind or rapidly changing cases remain difficult.

The [executed notebook](notebooks/modeling.ipynb) refits the selected models and
examines errors by realized wind strength, change magnitude, and year.
These are **post-hoc diagnostics on an already inspected test set**.
They are not used to retune models or presented as independent validation.

ERA5 is retrospective reanalysis; availability at forecast issuance has not been
established. Original timestamps lack explicit offsets. Their UTC interpretation
follows the earlier feature table but has not been independently checked against
original provider responses. These results therefore do not establish live forecasting quality.

## 10. Original project and current extension

The original project used a notebook with classical ML and TensorFlow/Keras models.
It was presented as a five-person course project, with most code implemented by
the repository developer; presentation assignments did not fully describe code ownership.

The current extension organizes the pipeline, aligns future targets, and adds
walk-forward validation, baselines, PyTorch, ablations, and result verification.
API/demo and Docker components have been removed to keep the repository focused
on modeling research.

The original and new metric tables are not directly comparable: the old ML branch
predicted the label at the same row, while the sequential branch predicted a later observation.

## 11. Reading the code

- **notebooks/modeling.ipynb:** current research narrative, executed tables, and figures.
- **src/:** reusable data, feature, splitting, model, evaluation, and analysis modules.
- **configs/research.json:** benchmark configuration.
- **reports/benchmark/:** published aggregate results and the benchmark figure.
- **reports/analysis/:** additional notebook diagnostics, not used for model selection.
- **tests/:** target alignment, leakage, artifact, and diagnostic checks.

Original observations, model weights, and observation-level predictions remain local.
After installing requirements.txt and providing the CSV, run:

    python main.py train --data data/raw/dataset.csv --config configs/research.json --output reports/runs/new-run --evaluate-holdout

Published benchmark results remain unchanged. Editing the notebook narrative is not
a new training experiment. These methods transfer to other time-series research,
but this project does not contain financial data or a trading backtest.
