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

## Citation

If you use this work, please cite the accompanying paper:

> Nishat, M. N. R. K. (2026). *Machine Learning for Historical Groundwater Arsenic Risk Assessment in Bangladesh: Spatial Validation, Explainability, and Uncertainty.*

## License

- **Code:** [MIT License](LICENSE) (edit if you'd prefer something else, e.g. Apache-2.0 or GPL-3.0).
- **Paper text and figures:** all rights reserved unless you choose to add a Creative Commons license.

## Author

**MD. Nahidur Rahman Khan Nishat** — Department of Computing and Information System, Daffodil International University.
