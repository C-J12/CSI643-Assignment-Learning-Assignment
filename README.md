# CSI643 Machine Learning Assignment

This project uses the UCI Student Performance Mathematics dataset to classify pass/fail outcomes and predict final grades (G3).

## Project structure

- `CSI643Asignment.ipynb` — original notebook uploaded by the student; preserved unchanged.
- `main.py` — runs the project pipeline.
- `src/data_loading.py` — downloads (if needed), loads, and inspects the dataset.
- `src/preprocessing.py` — shared numeric and categorical preprocessing.
- `src/classification.py` — C1 early and C2 grade-aware classification models.
- `src/regression.py` — R1 early and R2 grade-aware regression models.
- `src/evaluation.py` — plots and exports result tables.
- `requirements.txt` — Python dependencies.
- `data/` — location for `student-mat.csv` (downloaded automatically if absent).
- `results/` — generated charts and CSV result tables.

## Run locally

Use Python 3.10 or newer. From this folder, install dependencies and run:

```bash
pip install -r requirements.txt
python main.py
```

The script downloads the UCI dataset into `data/` if it is not already present. It creates plots and CSV evaluation tables in `results/`.

## Experiments

- Classification: Logistic Regression, Decision Tree, Random Forest; metrics include accuracy, precision, recall, F1 and ROC-AUC.
- Regression: Linear Regression, Decision Tree Regressor, Random Forest Regressor; metrics include MAE, RMSE and R².
- Early tasks exclude G1 and G2; grade-aware tasks include them but exclude G3 from predictors.

Note: the original notebook is retained as supplied and is not overwritten by the split Python files.
