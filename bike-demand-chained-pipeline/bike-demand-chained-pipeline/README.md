# Bike Demand — Chained ML Pipeline

VS Code-first implementation of GIU Machine Learning Project 1:
**Rush Hour: Predicting Bike-Sharing Demand — The Chained Pipeline Project**.

## Pipeline

1. Gradient Descent — linear regression from scratch
2. Polynomial Regression — using Phase 1 GD and weights as initialization
3. Bias–Variance Tradeoff — random vs chronological generalization diagnosis
4. Regularization — L1, L2 and Elastic Net
5. Logistic Regression — high-demand classification using Phase 4 survivors

## Repository layout

```text
bike-demand-chained-pipeline/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data.py
│   ├── features.py
│   ├── metrics.py
│   ├── visualization.py
│   ├── experiment_log.py
│   ├── phase1_gradient_descent.py
│   ├── phase2_polynomial.py
│   ├── phase3_bias_variance.py
│   ├── phase4_regularization.py
│   └── phase5_logistic.py
├── experiments/
│   ├── phase1/
│   ├── phase2/
│   ├── phase3/
│   ├── phase4/
│   └── phase5/
├── outputs/
│   ├── figures/
│   └── tables/
├── notebooks/
│   └── final_submission.ipynb
├── tests/
├── config.py
└── requirements.txt
```

## Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
pytest
```

## Ground rules

- Replace `TEAM_IDS` in `config.py` with the registered team roster.
- Fit scalers/encoders/feature-selection objects on the training portion only.
- Use validation for model decisions.
- Keep the hidden `test.csv` outside tuning.
- Every important modeling choice should have an experiment record and evidence.
- The final notebook is the presentation/submission layer; reusable logic lives in `src/`.
- Do not fill an `Outcome` section until its experiment has actually been run.

## Data

The supplied `train.csv`, `test.csv`, and `sample_submission.csv` are currently included
under `data/raw/` in this private-project scaffold.
