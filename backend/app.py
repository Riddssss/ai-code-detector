"""
AI vs Human Code Detector — Flask Backend API
=============================================
Endpoints:
  GET  /health          → health check
  POST /predict         → classify a code snippet
  GET  /predict?code=.. → classify via query param
"""
 
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import numpy as np
from scipy.sparse import hstack, csr_matrix
from pathlib import Path
import logging
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
 
# ─────────────────────────────────────────────
#  Artifact paths (adjust if needed)
# ─────────────────────────────────────────────
VECTORIZER_PATH = Path("tfidf_vectorizer.joblib")
SCALER_PATH     = Path("scaler.joblib")
LOGREG_PATH     = Path("trained_models/logreg_model.joblib")
RF_PATH         = Path("trained_models/rf_model.joblib")
 
# ─────────────────────────────────────────────
#  Load models at startup
# ─────────────────────────────────────────────
def load_artifacts():
    missing = [p for p in [VECTORIZER_PATH, SCALER_PATH, LOGREG_PATH] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing artifacts: {missing}. Run the training pipeline first."
        )
    vectorizer = joblib.load(VECTORIZER_PATH)
    scaler     = joblib.load(SCALER_PATH)
    logreg     = joblib.load(LOGREG_PATH)
    rf         = joblib.load(RF_PATH) if RF_PATH.exists() else None
    logger.info("All model artifacts loaded successfully.")
    return vectorizer, scaler, logreg, rf
 
vectorizer, scaler, logreg, rf = load_artifacts()
 
# ─────────────────────────────────────────────
#  Preprocessing helpers
# ─────────────────────────────────────────────
def preprocess_code(code: str) -> str:
    """Clean and normalise a code snippet."""
    if not code:
        return ""
    s = str(code)
    s = s.replace("\r\n", "\n").replace("\r", "\n").replace("\x0c", "\n")
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\n\s*\n+", "\n", s)
    s = s.replace("\t", "    ")
    s = re.sub(r"\b\d{10,}\b", "0", s)
    s = s.encode("utf-8", "replace").decode("utf-8")
    return s.strip()
 
 
def extract_stat_features(snippets: list[str]) -> np.ndarray:
    """Extract hand-crafted statistical features from code snippets."""
    rows = []
    for s in snippets:
        s = "" if s is None else str(s)
        lines = s.splitlines()
        num_lines = max(1, len(lines))
 
        avg_line_len = float(np.mean([len(ln) for ln in lines])) if lines else 0.0
        num_comments = sum(
            1 for ln in lines
            if ln.strip().startswith("#") or ln.strip().startswith("//") or "/*" in ln
        )
        comment_ratio      = num_comments / num_lines
        num_imports        = len(re.findall(r"\bimport\b|#include\b", s))
        num_defs           = len(re.findall(r"\bdef\b|\bclass\b|\bfunction\b", s))
        num_special_syms   = len(re.findall(r"[{}()<>;=+\-*/\\\[\]]", s))
        token_count        = len(re.findall(r"\w+", s))
        tokens             = re.findall(r"\w+", s)
        avg_token_len      = float(np.mean([len(t) for t in tokens])) if tokens else 0.0
 
        rows.append([
            avg_line_len, comment_ratio, num_imports, num_defs,
            num_special_syms, token_count, avg_token_len,
        ])
    return np.array(rows, dtype=float)
 
 
def build_feature_matrix(snippet: str):
    """Return the combined TF-IDF + statistical feature matrix for one snippet."""
    s       = preprocess_code(snippet)
    X_text  = vectorizer.transform([s])
    X_stats = scaler.transform(extract_stat_features([s]))
    return hstack([X_text, csr_matrix(X_stats)])
 
 
def predict_code(snippet: str, model_name: str = "logreg") -> dict:
    """Run inference and return label + probabilities."""
    model = rf if (model_name == "rf" and rf is not None) else logreg
    X     = build_feature_matrix(snippet)
    probs = model.predict_proba(X)[0]
    pred  = int(model.predict(X)[0])
    return {
        "label":       "ai" if pred == 1 else "human",
        "prob_ai":     round(float(probs[1]), 4),
        "prob_human":  round(float(probs[0]), 4),
        "model_used":  "random_forest" if (model_name == "rf" and rf) else "logistic_regression",
    }
 
# ─────────────────────────────────────────────
#  Flask application
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
 
 
@app.get("/health")
def health():
    return jsonify({"status": "ok", "models_loaded": True})
 
 
@app.route("/predict", methods=["POST", "GET"])
def predict_endpoint():
    """
    POST body: { "code": "...", "model": "logreg" | "rf" }
    GET  params: ?code=...&model=logreg
    """
    if request.method == "POST":
        body  = request.get_json(force=True, silent=True) or {}
        code  = body.get("code", "")
        model = body.get("model", "logreg")
    else:
        code  = request.args.get("code", "")
        model = request.args.get("model", "logreg")
 
    if not code or not code.strip():
        return jsonify({"error": "No code provided. Send 'code' in the request body or as a query param."}), 400
 
    try:
        result = predict_code(code, model_name=model)
        return jsonify(result)
    except Exception as exc:
        logger.exception("Prediction failed")
        return jsonify({"error": str(exc)}), 500
 
 
# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
 