
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score
)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 55)
print("  LOAN APPROVAL CLASSIFICATION PROJECT")
print("=" * 55)

df = pd.read_csv(r"C:\Users\jaffu\Desktop\Ml projects\test_Y3wMUE5_7gLdaTN.csv")

print(f"\n[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ─────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "─" * 55)
print("  STEP 1: EXPLORATORY DATA ANALYSIS")
print("─" * 55)

print("\n[Dataset Info]")
df.info()

print("\n[Missing Values per Column]")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"Missing": missing, "Percentage (%)": missing_pct})
print(missing_df[missing_df["Missing"] > 0])

print("\n[Basic Statistics - Numerical Columns]")
print(df.describe().round(2))

print("\n[Categorical Column Value Counts]")
cat_cols = df.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    if col != "Loan_ID":
        print(f"\n  >> {col}:")
        print(df[col].value_counts().to_string())

# ─────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────
print("\n" + "─" * 55)
print("  STEP 2: DATA PREPROCESSING")
print("─" * 55)

df_clean = df.copy()
df_clean.drop(columns=["Loan_ID"], inplace=True)

# Fill missing values
print("\n[Filling Missing Values...]")
df_clean["Gender"].fillna(df_clean["Gender"].mode()[0], inplace=True)
df_clean["Dependents"].fillna(df_clean["Dependents"].mode()[0], inplace=True)
df_clean["Self_Employed"].fillna(df_clean["Self_Employed"].mode()[0], inplace=True)
df_clean["LoanAmount"].fillna(df_clean["LoanAmount"].median(), inplace=True)
df_clean["Loan_Amount_Term"].fillna(df_clean["Loan_Amount_Term"].mode()[0], inplace=True)
df_clean["Credit_History"].fillna(df_clean["Credit_History"].mode()[0], inplace=True)

print("  ✓ Missing values filled successfully")
print(f"  ✓ Remaining nulls: {df_clean.isnull().sum().sum()}")

# Encode categorical columns
print("\n[Encoding Categorical Columns...]")
le = LabelEncoder()
encode_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
for col in encode_cols:
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
    print(f"  ✓ Encoded: {col}")

# ─────────────────────────────────────────────
# 4. SIMULATE TARGET LABEL (since test file has no Loan_Status)
#    We use Credit_History as a strong proxy for demo purposes.
# ─────────────────────────────────────────────
print("\n[NOTE] No 'Loan_Status' column found in test file.")
print("       Simulating target using Credit_History + LoanAmount for demo...")

np.random.seed(42)
df_clean["Loan_Status"] = (
    (df_clean["Credit_History"] == 1) &
    (df_clean["LoanAmount"] < df_clean["LoanAmount"].median() + 50)
).astype(int)

# Add slight randomness to make it realistic
noise = np.random.choice([0, 1], size=len(df_clean), p=[0.85, 0.15])
df_clean["Loan_Status"] = ((df_clean["Loan_Status"] + noise) > 0).astype(int)

print(f"\n[Simulated Target Distribution]")
counts = df_clean["Loan_Status"].value_counts()
print(f"  Approved (1): {counts.get(1, 0)} ({counts.get(1, 0)/len(df_clean)*100:.1f}%)")
print(f"  Rejected (0): {counts.get(0, 0)} ({counts.get(0, 0)/len(df_clean)*100:.1f}%)")

# ─────────────────────────────────────────────
# 5. TRAIN / TEST SPLIT & SCALING
# ─────────────────────────────────────────────
print("\n" + "─" * 55)
print("  STEP 3: TRAIN-TEST SPLIT & SCALING")
print("─" * 55)

X = df_clean.drop(columns=["Loan_Status"])
y = df_clean["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\n  Training samples : {len(X_train)}")
print(f"  Testing  samples : {len(X_test)}")
print(f"  Features         : {X.shape[1]}")

# ─────────────────────────────────────────────
# 6. MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────
print("\n" + "─" * 55)
print("  STEP 4: MODEL TRAINING & EVALUATION")
print("─" * 55)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    print(f"\n>>> Training: {name}")

    # Choose scaled or unscaled based on model type
    X_tr = X_train_scaled if name == "Logistic Regression" else X_train
    X_te = X_test_scaled  if name == "Logistic Regression" else X_test

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    acc      = accuracy_score(y_test, y_pred)
    roc_auc  = roc_auc_score(y_test, y_prob)
    cv_score = cross_val_score(model, X_tr, y_train, cv=5, scoring="accuracy").mean()

    results[name] = {"Accuracy": acc, "ROC-AUC": roc_auc, "CV Accuracy": cv_score}

    print(f"  ✓ Accuracy      : {acc*100:.2f}%")
    print(f"  ✓ ROC-AUC Score : {roc_auc:.4f}")
    print(f"  ✓ CV Accuracy   : {cv_score*100:.2f}% (5-fold)")

    print(f"\n  [Confusion Matrix - {name}]")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  {'':>15} Predicted 0  Predicted 1")
    print(f"  Actual 0   : {cm[0][0]:>8}     {cm[0][1]:>8}")
    print(f"  Actual 1   : {cm[1][0]:>8}     {cm[1][1]:>8}")

    print(f"\n  [Classification Report - {name}]")
    print(classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]))

# ─────────────────────────────────────────────
# 7. FEATURE IMPORTANCE (Random Forest)
# ─────────────────────────────────────────────
print("\n" + "─" * 55)
print("  STEP 5: FEATURE IMPORTANCE (Random Forest)")
print("─" * 55)

rf_model = models["Random Forest"]
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print("\n  Feature Importance Ranking:")
for i, (feat, imp) in enumerate(importances.items(), 1):
    bar = "█" * int(imp * 80)
    print(f"  {i:>2}. {feat:<22} {bar} {imp:.4f}")

# ─────────────────────────────────────────────
# 8. SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  FINAL SUMMARY")
print("=" * 55)

best_model = max(results, key=lambda k: results[k]["Accuracy"])
print(f"\n  Best Performing Model : {best_model}")
for k, v in results.items():
    print(f"\n  {k}:")
    print(f"    Accuracy    : {v['Accuracy']*100:.2f}%")
    print(f"    ROC-AUC     : {v['ROC-AUC']:.4f}")
    print(f"    CV Accuracy : {v['CV Accuracy']*100:.2f}%")

print("\n  [NEXT STEPS]")
print("  1. Load the actual TRAINING dataset with Loan_Status labels")
print("  2. Tune hyperparameters (GridSearchCV / RandomizedSearchCV)")
print("  3. Try XGBoost or Gradient Boosting for better accuracy")
print("  4. Handle class imbalance (SMOTE / class_weight='balanced')")
print("  5. Export the model using joblib for deployment")
print("\n" + "=" * 55)
print("  Pipeline Complete!")
print("=" * 55)
