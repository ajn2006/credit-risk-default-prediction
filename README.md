# credit-risk-default-prediction

Credit Risk / Loan Default Prediction Model

Predicting the probability that a loan will default, using historical loan-level data, with a focus on handling class imbalance correctly and evaluating the model the way real credit risk teams do — not just with accuracy.

Problem

Lenders need to estimate, at the point a loan is issued, how likely that borrower is to default. Getting this wrong in either direction is costly: reject too many good borrowers and you lose business; accept too many risky ones and you take on losses. This project builds a baseline default-prediction model and evaluates it with the metrics that actually matter for an imbalanced, decision-driving problem like this one.

Target variable: not.fully.paid (1 = loan was charged off / not fully repaid, 0 = fully paid).

Data


Source: publicly available LendingClub loan-level dataset (~9,500 rows, 14 columns).
Features available at loan origination: credit.policy, purpose, int.rate, installment, log.annual.inc, dti, fico, days.with.cr.line, revol.bal, revol.util, inq.last.6mths, delinq.2yrs, pub.rec.
Class balance: ~84% fully paid, ~16% default — a meaningful imbalance that shapes every modeling and evaluation decision below.


Methodology

Data cleaning


Categorical feature purpose one-hot encoded (pd.get_dummies, drop_first=True to avoid multicollinearity).
Verified no post-origination leakage in the feature set — all columns are known at the time the loan is issued (unlike the full raw LendingClub dataset, this subset excludes outcome-only fields such as payment history and recovery amounts).
credit.policy was kept as a feature but flagged as a judgment call, since it reflects the lender's own prior underwriting decision rather than a purely independent signal.


Modeling


Baseline model: logistic regression, chosen deliberately as a first step — it's the industry standard in credit scoring because it's interpretable and auditable, which matters given fair-lending regulatory requirements (e.g. ECOA) around explainability.
Features scaled with StandardScaler prior to fitting (required for logistic regression to converge reliably, and necessary for coefficients to be comparably interpretable).
Train/test split: 80/20, stratified on the target to preserve the true default rate in both sets given the class imbalance.


Evaluation

Accuracy is not informative here — a model predicting "no default" for every loan would score ~84% while being useless. Instead:


Precision-Recall curve: used instead of ROC, since ROC can look misleadingly strong on imbalanced data (false positive rate is diluted by the large non-default class). PR curve focuses specifically on performance on the minority (default) class, which is the class that matters operationally.
KS statistic (Kolmogorov-Smirnov): the industry-standard credit-scoring metric — the maximum separation between the cumulative distributions of predicted risk scores for actual defaulters vs. actual non-defaulters. Achieved KS = 0.25, within the "acceptable, workable" range (0.20–0.40) for a first-pass baseline model.
Calibration plot: checks whether predicted probabilities are trustworthy as actual probabilities — important since predicted PD feeds directly into expected loss calculations. The model calibrates well up to roughly 0.5 predicted probability; performance in the high-risk tail is noisier, which traces back to data sparsity (few loans in the dataset carry very high predicted risk), not necessarily a modeling flaw.


Results

MetricResultKS statistic0.25PR curvesee /plotsCalibrationwell-calibrated below ~0.5 predicted probability; noisier above, due to sparse high-risk samples

Limitations


Logistic regression baseline only — a gradient boosting model (XGBoost/LightGBM) would likely improve separation (higher KS) and is a natural next step, at the cost of interpretability.
Dataset is relatively small (~9,500 rows) and from a single lender/time period, so results may not generalize to other lending environments or economic conditions.
High-risk tail (predicted probability > ~0.6) is underrepresented in the data, making the model's calibration and reliability weaker in exactly the range that matters most for flagging genuinely risky loans.
No transaction costs, recovery rates (LGD), or exposure-at-default data included yet — expected loss (PD × LGD × EAD) has not been calculated in this version.


Next steps


Compare logistic regression against XGBoost/LightGBM and report the interpretability-vs-performance trade-off explicitly.
Add SHAP values for feature-level interpretability and a fair-lending sanity check (verify no feature proxies for a protected characteristic).
Extend to an expected-loss and threshold/cutoff analysis to translate model output into an actual lending decision framework.
Replace the single train/test split with stratified k-fold cross-validation to reduce sensitivity to any one split, given how small the minority class is.
