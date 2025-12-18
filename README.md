# Credit Default Risk Prediction

## Problem Statement
Financial institutions need to assess credit risk accurately to minimize loan defaults.
This project predicts the probability of loan default using borrower demographic and
credit history data.

## Dataset
Public credit risk dataset with ~32,000 records and ~22% default rate.

## Modeling Approach
- Logistic Regression (baseline)
- XGBoost
- LightGBM (final model)

## Evaluation Metrics
- ROC-AUC
- Recall for defaulters (class 1)

## Results
| Model | ROC-AUC |
|------|--------|
| Logistic Regression | 0.87 |
| XGBoost | 0.93 |
| LightGBM | **0.95** |

## Final Model
LightGBM was selected due to superior discrimination power, robustness to feature scaling,
and better capture of non-linear risk patterns.

## Streamlit Application
This project includes a Streamlit app for real-time credit default prediction.

### Run Locally
### Run Locally
### Run Locally
'''bash
pip install -r requirements.txt
streamlit run app/app.py
'''

