# Predicting Loan Default on the Home Credit Dataset

[![Kaggle competition](https://img.shields.io/badge/Kaggle-Home%20Credit%20Default%20Risk-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/home-credit-default-risk)
![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Course project for **Artificial Intelligence & Machine Learning (623.504)**, winter semester 2026/27
([course materials](https://github.com/konstantin-schekotihin/dl_course/tree/master/AI-ML) ·
[project guide](https://github.com/konstantin-schekotihin/dl_course/tree/master/AI-ML/deliverables)).

We predict whether a consumer-loan applicant will have payment difficulties, using the
[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) dataset from Kaggle,
and compare three model families that sit at different points of the accuracy-vs-interpretability trade-off:

| Model family | Role |
| --- | --- |
| Regularized logistic regression (L1 / ElasticNet) | Interpretable, scorecard-style baseline |
| Gradient-boosted trees (random forest as comparison) | Expected strongest model on tabular data |
| Multi-layer perceptron (MLP) | Neural-network contender |

## Motivation

Lending is a prediction problem: approving an applicant who later defaults loses a large part of the loan,
while rejecting a reliable applicant loses the interest income. This matters most for people with little or
no credit history, which is the group Home Credit serves.

Credit scoring is classified as a **high-risk application under the EU AI Act**, and banks have traditionally
relied on logistic-regression scorecards because every decision can be traced back to individual features.
This project asks:

> *How much predictive performance do non-linear models gain over an interpretable linear model, and is this
> gain worth the loss of interpretability?*

**Hypothesis.** Gradient-boosted trees perform best, a regularized logistic regression is only slightly worse,
and an MLP is competitive without outperforming the trees on this tabular data.

## Learning task (T, P, E)

- **Task T: binary classification.** For each application $x_n$, the model outputs the probability
  $y_n = P(t_n = 1 \mid x_n)$ that the client has payment difficulties ($t_n = 1$: at least one of the first
  installments paid late by more than a defined number of days). An application is flagged as risky if
  $y_n \ge \tau$, where the threshold $\tau$ is chosen after training.
- **Performance measure P.** Only about 8% of applications are defaults, so accuracy is misleading.
  - **ROC-AUC** (primary, and the official competition metric)
  - **Expected cost** $\mathrm{Cost}(\tau) = c_{FN} \cdot FN(\tau) + c_{FP} \cdot FP(\tau)$, with a sensitivity analysis over the unknown cost ratio $c_{FN}/c_{FP}$
  - **Precision, recall, F1** at the cost-optimal threshold and at a fixed rejection rate (e.g. the riskiest 10%)
  - **PR-AUC**, **Brier score** (calibration), and **training / inference runtime**
- **Experience E.** `application_train` with about 307,500 labeled applications and about 120 features,
  plus `application_test` (about 48,700 applications, hidden labels) scored through a Kaggle submission.
  Six additional tables (credit bureau history, previous applications, installment payments, …) are linked via the client ID.

## Approach

1. **Data pipeline and EDA.** Class balance, missing values (several columns are more than 50% missing),
   anomalies such as the `DAYS_EMPLOYED = 365243` placeholder, and domain features like credit-to-income,
   annuity-to-income, loan term, and age in years.
2. **Preprocessing.** Imputation with missing-value indicators, scaling, and one-hot encoding, combined in
   `sklearn` pipelines that are fitted on training data only. Models are trained on the original class
   distribution; class weights are treated as a hyperparameter.
3. **Validation and tuning.** A stratified 20% hold-out of `application_train` serves as our own test set.
   Hyperparameters are tuned with 5-fold cross-validation on the remaining 80% using
   [Weights & Biases](https://wandb.ai/) sweeps, with fixed random seeds for reproducibility.
4. **Evaluation.** One comparison table across all metrics, an error analysis of misclassified applicants,
   and a final score on the Kaggle test set.
5. **Extension.** Aggregated features from the additional tables (e.g. number of previous loans, share of
   refused applications) and how much each model family gains from them.

### Baselines

| Reference | ROC-AUC |
| --- | --- |
| Majority-class `DummyClassifier` | 0.50 |
| Logistic regression on `EXT_SOURCE_1/2/3` only | to be measured |
| Kaggle winners (all tables, thousands of features, GBDT ensembles) | ≈ 0.80 |

Because this project focuses on the main table plus a few aggregated features, we expect scores somewhat below the top of the leaderboard.

## Repository layout

```text
.
├── data/          # Kaggle data goes here (not committed, see data/README.md)
├── notebooks/     # EDA and experiment notebooks
├── src/           # Shared preprocessing, feature engineering, and evaluation code
├── reports/       # Figures and the final report
└── requirements.txt
```

## Getting started

```bash
git clone https://github.com/alexschnabl/AI-ML-26W.git
cd AI-ML-26W
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then download the competition data as described in [`data/README.md`](data/README.md).
You need a Kaggle account and must accept the competition rules before the data can be downloaded.

## Course

- [Course materials](https://github.com/konstantin-schekotihin/dl_course/tree/master/AI-ML): lecture notebooks (classifiers, regression, logistic regression, neural networks, ensembles, SVMs, unsupervised learning)
- [Project guide and deliverables](https://github.com/konstantin-schekotihin/dl_course/tree/master/AI-ML/deliverables): how to prepare the proposal, the report and the oral defense, with LaTeX templates
- [Course modalities](https://github.com/konstantin-schekotihin/dl_course/blob/master/AI-ML/modalities.md): teams, grading, deadlines and W&B tracking

## References

- A. Montoya, inversion, K. Odintsov, M. Kotek. [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk). Kaggle competition, 2018.
- deepsense.ai. [Wait, so loans need to be repaid? The Home Credit risk prediction competition on Kaggle](https://deepsense.ai/wait-so-loans-need-to-be-repaid-the-home-credit-risk-prediction-competition-on-kaggle/), 2018.
- S. Lessmann, B. Baesens, H.-V. Seow, L. C. Thomas. [Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research](https://doi.org/10.1016/j.ejor.2015.05.030). *European Journal of Operational Research* 247(1), 2015.
- X. Dastile, T. Celik, M. Potsane. [Statistical and machine learning models in credit scoring: A systematic literature survey](https://doi.org/10.1016/j.asoc.2020.106263). *Applied Soft Computing* 91, 2020.
- L. Grinsztajn, E. Oyallon, G. Varoquaux. [Why do tree-based models still outperform deep learning on typical tabular data?](https://arxiv.org/abs/2207.08815) NeurIPS Datasets and Benchmarks, 2022.
- R. Shwartz-Ziv, A. Armon. [Tabular data: Deep learning is not all you need](https://arxiv.org/abs/2106.03253). *Information Fusion* 81, 2022.
- Y. Gorishniy, I. Rubachev, V. Khrulkov, A. Babenko. [Revisiting deep learning models for tabular data](https://arxiv.org/abs/2106.11959). NeurIPS, 2021.
- G. Ke et al. [LightGBM: A highly efficient gradient boosting decision tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree). NeurIPS, 2017.
- T. Chen, C. Guestrin. [XGBoost: A scalable tree boosting system](https://arxiv.org/abs/1603.02754). KDD, 2016.
- T. M. Mitchell. *Machine Learning*. McGraw-Hill, 1997.

## License

Code in this repository is released under the [MIT License](LICENSE).
The Home Credit data is subject to the [Kaggle competition rules](https://www.kaggle.com/competitions/home-credit-default-risk/rules) and is not redistributed here.
