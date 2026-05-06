"""
=======================================================
  LOAN APPROVAL — FULL MODEL TRAINING PIPELINE
=======================================================
  Steps:
    1. Generate realistic labeled training data
    2. Preprocess (impute, encode, scale)
    3. Train & compare 4 models with cross-validation
    4. Evaluate best model (classification report + ROC AUC)
    5. Save model.pkl + scaler.pkl → api/
=======================================================
"""

import os, warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection        import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing          import LabelEncoder, StandardScaler
from sklearn.linear_model           import LogisticRegression
from sklearn.ensemble               import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics                import (classification_report,
                                            confusion_matrix,
                                            roc_auc_score,
                                            accuracy_score)
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
API_DIR     = os.path.join(BASE_DIR, "api")
MODEL_PATH  = os.path.join(API_DIR, "model.pkl")
SCALER_PATH = os.path.join(API_DIR, "scaler.pkl")
os.makedirs(API_DIR, exist_ok=True)

print("=" * 60)
print("  LOAN APPROVAL — MODEL TRAINING PIPELINE")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# STEP 1 ▸ Generate Realistic Labeled Training Data
# ─────────────────────────────────────────────────────────────
print("\n[STEP 1] Generating realistic training dataset…")

np.random.seed(42)
N = 614   # matches real Analytics Vidhya dataset size

gender        = np.random.choice(["Male","Female"],           N, p=[0.813, 0.187])
married       = np.random.choice(["Yes","No"],                N, p=[0.651, 0.349])
dependents    = np.random.choice(["0","1","2","3+"],          N, p=[0.545, 0.162, 0.164, 0.129])
education     = np.random.choice(["Graduate","Not Graduate"], N, p=[0.782, 0.218])
self_employed = np.random.choice(["No","Yes"],                N, p=[0.857, 0.143])
property_area = np.random.choice(["Urban","Semiurban","Rural"],N,p=[0.382, 0.340, 0.278])

# Correlated numerical features
applicant_income    = np.random.lognormal(8.4, 0.7, N).astype(int)   # ~5000 median
coapplicant_income  = np.where(married == "Yes",
                                np.random.lognormal(7.2, 0.8, N).astype(int),
                                np.zeros(N, dtype=int))
loan_amount         = np.clip(
    (applicant_income + coapplicant_income) * 0.018 +
    np.random.normal(0, 20, N), 28, 700).astype(int)
loan_amount_term    = np.random.choice([12,36,60,84,120,180,240,300,360,480],
                                        N, p=[0.01,0.01,0.02,0.01,0.05,0.03,0.04,0.01,0.79,0.03])

# Credit history with slight randomness
credit_good  = np.random.random(N) < 0.84
credit_history = credit_good.astype(float)

# ── Realistic Loan_Status logic ────────────────────────────────
# Base approval probability
base   = 0.40
prob   = np.full(N, base)

prob  += (credit_history == 1)           * 0.38   # strong positive
prob  += (education == "Graduate")       * 0.07
prob  += (married == "Yes")              * 0.05
prob  += (property_area == "Semiurban")  * 0.05
prob  += (property_area == "Urban")      * 0.03
prob  -= (self_employed == "Yes")        * 0.04
prob  -= (dependents == "3+")            * 0.06
income_ratio = (applicant_income + coapplicant_income) / (loan_amount * 1000 + 1)
prob  += np.clip(income_ratio * 0.3, -0.1, 0.15)
prob   = np.clip(prob, 0.02, 0.98)

loan_status = (np.random.random(N) < prob).astype(int)   # 1=Approved, 0=Rejected

df = pd.DataFrame({
    "Gender":            gender,
    "Married":           married,
    "Dependents":        dependents,
    "Education":         education,
    "Self_Employed":     self_employed,
    "ApplicantIncome":   applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount":        loan_amount,
    "Loan_Amount_Term":  loan_amount_term,
    "Credit_History":    credit_history,
    "Property_Area":     property_area,
    "Loan_Status":       loan_status,
})

print(f"  ✓ Dataset: {len(df)} rows, {df.shape[1]} columns")
print(f"  ✓ Approval rate: {loan_status.mean()*100:.1f}%  "
      f"(Approved={loan_status.sum()}, Rejected={N - loan_status.sum()})")

# ─────────────────────────────────────────────────────────────
# STEP 2 ▸ Preprocessing
# ─────────────────────────────────────────────────────────────
print("\n[STEP 2] Preprocessing…")

le = LabelEncoder()
encode_cols = ["Gender","Married","Dependents","Education","Self_Employed","Property_Area"]
for col in encode_cols:
    df[col] = le.fit_transform(df[col].astype(str))
    print(f"  ✓ Encoded: {col}")

# Store encoding maps for Flask API (same as LabelEncoder alphabetical order)
ENCODE_MAP = {
    "Gender":        {"Female": 0, "Male": 1},
    "Married":       {"No": 0, "Yes": 1},
    "Dependents":    {"0": 0, "1": 1, "2": 2, "3+": 3},
    "Education":     {"Graduate": 0, "Not Graduate": 1},
    "Self_Employed": {"No": 0, "Yes": 1},
    "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2},
}

X = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler   = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"  ✓ Train: {len(X_train)}, Test: {len(X_test)}")

# ─────────────────────────────────────────────────────────────
# STEP 3 ▸ Train & Compare 4 Models
# ─────────────────────────────────────────────────────────────
print("\n[STEP 3] Training & comparing models (5-fold CV)…")
print(f"  {'Model':<28} {'CV Accuracy':>12}  {'CV Std':>8}")
print("  " + "─" * 52)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
candidates = {
    "Logistic Regression":    LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":          RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting":      GradientBoostingClassifier(n_estimators=200, random_state=42),
}

# Try importing XGBoost (optional)
try:
    from xgboost import XGBClassifier
    candidates["XGBoost"] = XGBClassifier(n_estimators=200, random_state=42,
                                           use_label_encoder=False, eval_metric="logloss",
                                           verbosity=0)
except ImportError:
    print("  ⚠  XGBoost not installed — skipping (pip install xgboost to add)")

results = {}
for name, clf in candidates.items():
    scores = cross_val_score(clf, X_train_s, y_train, cv=cv, scoring="accuracy")
    results[name] = {"model": clf, "mean": scores.mean(), "std": scores.std()}
    print(f"  {name:<28} {scores.mean()*100:>10.2f}%  ±{scores.std()*100:.2f}%")

best_name = max(results, key=lambda k: results[k]["mean"])
best_clf  = results[best_name]["model"]
print(f"\n  🏆 Best model: {best_name} ({results[best_name]['mean']*100:.2f}%)")

# ─────────────────────────────────────────────────────────────
# STEP 4 ▸ Evaluate Best Model on Test Set
# ─────────────────────────────────────────────────────────────
print(f"\n[STEP 4] Final evaluation of '{best_name}' on held-out test set…")

best_clf.fit(X_train_s, y_train)
y_pred  = best_clf.predict(X_test_s)
y_proba = best_clf.predict_proba(X_test_s)[:, 1]

acc     = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
cm      = confusion_matrix(y_test, y_pred)

print(f"\n  Accuracy : {acc*100:.2f}%")
print(f"  ROC AUC  : {roc_auc:.4f}")
print("\n  Confusion Matrix:")
print(f"    {'':12} Pred:Rejected  Pred:Approved")
print(f"    True:Rejected  {cm[0][0]:^13}  {cm[0][1]:^13}")
print(f"    True:Approved  {cm[1][0]:^13}  {cm[1][1]:^13}")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred,
                              target_names=["Rejected","Approved"],
                              digits=3))

# Feature importance
if hasattr(best_clf, "feature_importances_"):
    importance = pd.Series(best_clf.feature_importances_, index=X.columns)
    importance = importance.sort_values(ascending=False)
    print("  Top Feature Importances:")
    for feat, imp in importance.head(6).items():
        bar = "█" * int(imp * 50)
        print(f"    {feat:<22} {imp:.3f}  {bar}")

# ─────────────────────────────────────────────────────────────
# STEP 5 ▸ Save Model + Scaler
# ─────────────────────────────────────────────────────────────
print(f"\n[STEP 5] Saving model to {API_DIR}…")

# Retrain on FULL dataset for best production performance
best_clf.fit(scaler.fit_transform(X), y)
joblib.dump(best_clf, MODEL_PATH)
joblib.dump(scaler,   SCALER_PATH)

print(f"  ✓ model.pkl  → {MODEL_PATH}")
print(f"  ✓ scaler.pkl → {SCALER_PATH}")
print(f"\n{'='*60}")
print(f"  ✅ Pipeline complete! Flask API ready to serve predictions.")
print(f"     Run: python api/app.py")
print(f"{'='*60}\n")
