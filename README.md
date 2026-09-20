# Historical Groundwater Arsenic Risk Assessment in Bangladesh

Machine learning analysis of the 1998–1999 National Hydrochemical Survey of Bangladesh, testing whether spatially validated, explainable models can characterize historical arsenic risk from tube-well chemistry and location data.

> [!WARNING]
> **Historical data only.** This project analyzes a 27-year-old survey for methodological research. It does **not** assess present-day water safety and cannot certify that any individual well is safe to drink from today.

## What this project does

- Cleans and explores the 1998–99 National Hydrochemical Survey (3,534 wells, 20 measured elements).
- Frames two prediction setups: **Scenario A** (full chemistry + well details — explanatory) and **Scenario B** (location + well details only — screening, usable on an untested well).
- Trains and compares five models — Random Forest, XGBoost, LightGBM, CatBoost, and a small MLP — against dummy/logistic/ridge baselines, for both classifying arsenic exceedance (>50 µg/L) and regressing log-arsenic concentration.
- Validates under three schemes — random split, single spatial-block holdout, and an eight-times-repeated spatial-block holdout — to check whether accuracy survives geographically new areas.
- Explains the best classifier with **SHAP**.
- Quantifies regression uncertainty with **conformalized quantile regression** (90% intervals), checked against a bootstrap baseline.
- Builds an exploratory multi-contaminant (As/Fe/Mn) composite risk score and tests sensitivity to the WHO vs. national arsenic guideline.

Full write-up, methodology, and results: [`paper/groundwater_paper.pdf`](paper/groundwater_paper.pdf).

## Data

- **Source:** British Geological Survey & Department of Public Health Engineering (2001). *Arsenic contamination of groundwater in Bangladesh* (Vol. 1: Summary). BGS Technical Report WC/00/19.
- **Dataset (Kaggle mirror):** [Bangladesh National Groundwater Hydrochemical](https://www.kaggle.com/datasets/mdnahidurrahmankh/bangladesh-national-groundwater-hydrochemical)
- **Working file:** `NationalSurveyData.csv`, accessed via the Kaggle dataset above.
- The raw CSV is **not included** in this repository (check the Kaggle dataset's license before redistributing it yourself). Download it from the link above and place it at `data/NationalSurveyData.csv` to rerun the notebook.

## Model architecture

<p align="center">
  <img src="https://i.postimg.cc/y6nNT1kr/Untitled-Diagram-drawio-(2).png" width="100%">
</p>

<p align="center"><em>Figure 1. End-to-end training and analysis pipeline.</em></p>

The pipeline runs left to right in five stages:

1. **Data preparation.** The 1998–99 survey is cleaned and quality-checked (below-detection-limit values handled, duplicates removed, implausible coordinates and well depths set to missing), then turned into model-ready features.
2. **Model training.** Two prediction setups are built: **Scenario A** (full chemistry + well details, explanatory) and **Scenario B** (location + well details, screening). Five models are compared for both classification (arsenic above 50 µg/L) and regression (log₁₀ arsenic): Random Forest, XGBoost, LightGBM, CatBoost, and a small MLP.
3. **Validation.** Each model is scored under three schemes: a random split, a single spatial-block holdout, and eight repeated spatial-block holdouts. The repeated spatial mean ± SD is the headline result, since it shows how well a model transfers to geographically new areas.
4. **Interpretation & uncertainty.** The best model is selected by mean repeated-spatial ROC-AUC. It is explained with **SHAP**, and its regression uncertainty is quantified with **conformalized quantile regression** (90% intervals).
5. **Risk outputs.** Predictions are mapped as historical arsenic risk, followed by an exploratory multi-contaminant (As / Fe / Mn) analysis.

> [!NOTE]
> The pipeline describes a methodological study of a 1998–99 survey. Its outputs are historical and should not be used to judge the safety of any well today.

## Analysis notebook

All calculations, models, figures, and numbers in the paper come from a single Jupyter notebook: [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb). It runs top to bottom with a fixed random seed (`42`) and finds the survey CSV automatically (it looks in `/kaggle/input` first, then the working directory), so it works both on Kaggle and locally.

### What the notebook does

| Step | What happens |
| ---- | ------------ |
| **Load & clean** | Detects the header row, maps column names to a standard set (coordinates, well depth, install year, well type, location, and the measured chemistry variables), and reads units from the headers. Values reported as below detection limit (`<x`) are set to half the limit and flagged. Duplicates are dropped; coordinates outside Bangladesh, implausible well depths (≤ 0 or > 500 m), and install years outside 1930–1999 are set to missing. |
| **Explore** | Well coverage map, distributions of As / Fe / Mn / F / NO₃-N / pH (log scale where skewed), element vs. depth plots, and an arsenic map. |
| **Targets & guidelines** | Flags wells exceeding Bangladesh national guidelines (As 50 µg/L, Mn 0.1 mg/L, Fe 1.0 mg/L, F 1.0 mg/L, NO₃-N 10 mg/L) with WHO reference values alongside. Arsenic is the modelling target: classification (exceeds 50 µg/L) and regression (log₁₀ concentration). |
| **Scenarios** | **Scenario A** uses water chemistry + well details (explanatory). **Scenario B** uses coordinates + well details + division/district only (screening, usable before sampling). |
| **Spatial blocks** | Wells are grouped into a ~0.25° (~25 km) grid so that nearby wells are never split between training and testing. A sensitivity check repeats validation at half and double the block size. |
| **Validation** | Compares a random 80/20 split, a single spatial-block holdout, and **8 repeated spatial-block holdouts**; the repeated spatial mean ± SD is the headline number. |
| **Models** | Dummy, Logistic/Ridge baselines, Random Forest, XGBoost, LightGBM, CatBoost, and a small MLP (one hidden layer of 32 units). Tree models are tuned with randomized search (12 iterations) using spatially grouped cross-validation. |
| **Metrics** | Classification: ROC-AUC, PR-AUC, F1, precision, recall, specificity (at a screening threshold set to the observed exceedance rate). Regression: MAE, RMSE, R², Spearman. |
| **Explainability** | SHAP summary and dependence plots for the best classifier, plus a ranked list of mean absolute SHAP values. |
| **Uncertainty** | 90% prediction intervals from conformalized quantile regression (CQR), compared against a bootstrap + residual baseline on the same held-out wells. |
| **Multi-contaminant score** | Exploratory composite (As / Fe / Mn and any other eligible contaminant) built from per-contaminant Scenario B logistic models, with a calibration (Brier score) check. Labelled exploratory throughout. |
| **Guideline sensitivity** | Re-scores the same arsenic model against the WHO (10 µg/L) and national (50 µg/L) thresholds, including how many held-out wells fall between the two. |
| **Maps & summary** | Observed vs. out-of-fold predicted exceedance maps, composite-risk and interval-width maps, and a printed results summary. |

### Notebook requirements

`numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `shap`, and `jupyter`. The XGBoost, LightGBM, CatBoost, and SHAP steps are skipped with a message if a library is missing, so the notebook still runs with a reduced set of models. Runtime is dominated by the hyperparameter search and the repeated spatial holdouts.

## Key results

See the paper for full detail.

**ROC-AUC under repeated spatial holdout (mean, SD)**

| Model         | ROC-AUC (SD)  |
| ------------- | ------------- |
| CatBoost      | 0.953 (0.009) |
| XGBoost       | 0.953 (0.008) |
| LightGBM      | 0.950 (0.009) |
| Random Forest | 0.950 (0.009) |
| MLP           | 0.893 (0.039) |

- Random-split accuracy did not meaningfully overstate spatially validated accuracy (difference ≈ 0.001).
- SHAP: phosphorus, iron, and strontium were the classifier's most influential variables.
- Conformal 90% intervals covered 91.4% of held-out wells, but were wide (median width ≈ 2 log<sub>10</sub> units).
- The combined multi-contaminant score was dominated by manganese and did not usefully separate wells.
- Switching from the national standard (50 µg/L) to the WHO guideline (10 µg/L) raised the exceeding share from 24.9% to 42.1%.

## Reproducing

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Download the dataset from Kaggle and place it at data/NationalSurveyData.csv
jupyter notebook notebooks/analysis.ipynb
```

Then run the notebook from top to bottom (**Run All**). It locates the CSV automatically, so no path needs editing. To run it on Kaggle instead, add the [dataset](https://www.kaggle.com/datasets/mdnahidurrahmankh/bangladesh-national-groundwater-hydrochemical) to a Kaggle notebook and upload `analysis.ipynb`.

## Citation

If you use this work, please cite the accompanying paper:

> Nishat, M. N. R. K. (2026). *Machine Learning for Historical Groundwater Arsenic Risk Assessment in Bangladesh: Spatial Validation, Explainability, and Uncertainty.*

## License

- **Code:** [MIT License](LICENSE) (edit if you'd prefer something else, e.g. Apache-2.0 or GPL-3.0).
- **Paper text and figures:** all rights reserved unless you choose to add a Creative Commons license.

## Author

**MD. Nahidur Rahman Khan Nishat** — Department of Computing and Information System, Daffodil International University.
