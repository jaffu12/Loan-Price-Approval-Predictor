from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

app = Flask(__name__)
CORS(app)  # Allow the frontend index.html to call this API

# ── Paths ────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# CSV is one level up from api/
CSV_PATH = os.path.join(BASE_DIR, "..", "test_Y3wMUE5_7gLdaTN.csv")

# ── Train & save model ───────────────────────────────────────
def train_and_save():
    df = pd.read_csv(CSV_PATH)
    df = df.drop(columns=["Loan_ID"])

    # Fill missing values
    for col in ["Gender", "Dependents", "Self_Employed"]:
        df = df.assign(**{col: df[col].fillna(df[col].mode()[0])})
    df = df.assign(
        LoanAmount       = df["LoanAmount"].fillna(df["LoanAmount"].median()),
        Loan_Amount_Term = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0]),
        Credit_History   = df["Credit_History"].fillna(df["Credit_History"].mode()[0]),
    )

    # Encode categoricals
    le = LabelEncoder()
    for col in ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]:
        df = df.assign(**{col: le.fit_transform(df[col].astype(str))})

    # Simulate target (test CSV has no Loan_Status column)
    np.random.seed(42)
    approved = ((df["Credit_History"] == 1) & (df["LoanAmount"] < df["LoanAmount"].median() + 50)).astype(int)
    noise    = np.random.choice([0, 1], size=len(df), p=[0.85, 0.15])
    df = df.assign(Loan_Status=((approved + noise) > 0).astype(int))

    X = df.drop(columns=["Loan_Status"])
    y = df["Loan_Status"]

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("[API] Model trained and saved.")
    return model, scaler

# ── Load or train model on startup ──────────────────────────
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("[API] Loaded existing model.")
else:
    model, scaler = train_and_save()

# ── Encoding maps (must mirror LabelEncoder order) ───────────
ENCODE_MAP = {
    "Gender":        {"Female": 0, "Male": 1},
    "Married":       {"No": 0, "Yes": 1},
    "Dependents":    {"0": 0, "1": 1, "2": 2, "3+": 3},
    "Education":     {"Graduate": 0, "Not Graduate": 1},
    "Self_Employed": {"No": 0, "Yes": 1},
    "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2},
}

# ── Routes ───────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "Loan Approval API is running ✅"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        features = [
            ENCODE_MAP["Gender"].get(data.get("gender", "Male"), 1),
            ENCODE_MAP["Married"].get(data.get("married", "No"), 0),
            ENCODE_MAP["Dependents"].get(str(data.get("dependents", "0")), 0),
            ENCODE_MAP["Education"].get(data.get("education", "Graduate"), 0),
            ENCODE_MAP["Self_Employed"].get(data.get("selfEmployed", "No"), 0),
            float(data.get("applicantIncome", 0)),
            float(data.get("coapplicantIncome", 0)),
            float(data.get("loanAmount", 0)),
            float(data.get("loanAmountTerm", 360)),
            float(data.get("creditHistory", 1)),
            ENCODE_MAP["Property_Area"].get(data.get("propertyArea", "Urban"), 2),
        ]

        X_input  = np.array(features).reshape(1, -1)
        X_scaled = scaler.transform(X_input)

        prediction  = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0]

        return jsonify({
            "prediction":    int(prediction),
            "label":         "Approved" if prediction == 1 else "Rejected",
            "confidence":    round(float(max(probability)) * 100, 2),
            "approved_prob": round(float(probability[1]) * 100, 2),
            "rejected_prob": round(float(probability[0]) * 100, 2),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
