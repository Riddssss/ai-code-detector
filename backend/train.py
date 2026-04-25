"""
AI vs Human Code Detector — Training Pipeline
==============================================
Run this script to prepare data and train models before starting the API.
 
Usage:
  python train.py --ai_pdf "ai codes.pdf" --human_pdf "human code.pdf"
"""
 
import argparse
import logging
import os
import re
from pathlib import Path
 
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
 
# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
OUT_CSV      = "code_dataset.csv"
VECT_PATH    = "tfidf_vectorizer.joblib"
SCALER_PATH  = "scaler.joblib"
OUT_DIR      = Path("trained_models")
 
# ─────────────────────────────────────────────
#  Step 1: PDF ingestion
# ─────────────────────────────────────────────
def extract_pages_text(pdf_path: str) -> list[str]:
    """Extract page text from a PDF using pdfplumber (falls back to PyPDF2)."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            texts = [p.extract_text() or "" for p in pdf.pages]
        logger.info(f"[pdfplumber] {path.name} → {len(texts)} pages")
        return texts
    except Exception:
        import PyPDF2
        texts = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for p in reader.pages:
                try:
                    texts.append(p.extract_text() or "")
                except Exception:
                    texts.append("")
        logger.info(f"[PyPDF2] {path.name} → {len(texts)} pages")
        return texts
 
 
def split_into_code_snippets(page_text: str) -> list[str]:
    """Heuristically extract code snippets from page text."""
    snippets = []
    if not page_text or not page_text.strip():
        return snippets
    parts = re.split(r"```+", page_text)
    if len(parts) > 1:
        for i in range(1, len(parts), 2):
            s = parts[i].strip()
            if len(s) > 20:
                snippets.append(s)
    else:
        blocks = [b.strip() for b in re.split(r"\n{2,}", page_text) if b.strip()]
        for block in blocks:
            if re.search(r"[;{}()=<>:]|def |class |import |#include|//", block) \
                    or len(block.splitlines()) > 6:
                if len(block) > 20:
                    snippets.append(block)
    return snippets
 
 
def preprocess_code(code: str) -> str:
    """Normalise a raw code snippet."""
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
 
 
def build_dataset(ai_pdf: str, human_pdf: str) -> pd.DataFrame:
    """Extract, clean, and balance snippets from both PDFs."""
    def _collect(pdf):
        snippets = []
        for page in extract_pages_text(pdf):
            snippets.extend(split_into_code_snippets(page))
        return [preprocess_code(s) for s in snippets if preprocess_code(s)]
 
    ai_snips    = _collect(ai_pdf)
    human_snips = _collect(human_pdf)
    logger.info(f"Raw snippets → AI: {len(ai_snips)}, Human: {len(human_snips)}")
 
    n           = min(len(ai_snips), len(human_snips))
    codes       = ai_snips[:n] + human_snips[:n]
    labels      = [1] * n + [0] * n
 
    df = pd.DataFrame({"code": codes, "label": labels})
    df.to_csv(OUT_CSV, index=False, encoding="utf-8", errors="replace")
    logger.info(f"Dataset saved → {OUT_CSV}  (rows={len(df)})")
    return df
 
# ─────────────────────────────────────────────
#  Step 2: Feature extraction
# ─────────────────────────────────────────────
def extract_stat_features(snippets: list[str]) -> pd.DataFrame:
    """Return a DataFrame of hand-crafted statistical features."""
    rows = []
    for s in snippets:
        s = "" if s is None else str(s)
        lines      = s.splitlines()
        num_lines  = max(1, len(lines))
        tokens     = re.findall(r"\w+", s)
        rows.append({
            "avg_line_len":       float(np.mean([len(ln) for ln in lines])) if lines else 0.0,
            "comment_ratio":      sum(
                1 for ln in lines
                if ln.strip().startswith("#") or ln.strip().startswith("//") or "/*" in ln
            ) / num_lines,
            "num_imports":        len(re.findall(r"\bimport\b|#include\b", s)),
            "num_defs":           len(re.findall(r"\bdef\b|\bclass\b|\bfunction\b", s)),
            "num_special_syms":   len(re.findall(r"[{}()<>;=+\-*/\\\[\]]", s)),
            "token_count":        len(tokens),
            "avg_token_len":      float(np.mean([len(t) for t in tokens])) if tokens else 0.0,
        })
    return pd.DataFrame(rows)
 
 
def build_features(df: pd.DataFrame, fit: bool = True):
    """Fit (or load) vectorizer/scaler and return (X, y, vectorizer, scaler)."""
    codes = df["code"].astype(str).tolist()
 
    # TF-IDF
    if not fit and Path(VECT_PATH).exists():
        vectorizer = joblib.load(VECT_PATH)
        logger.info(f"Loaded vectorizer from {VECT_PATH}")
    else:
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 6), max_features=20_000)
        vectorizer.fit(codes)
        joblib.dump(vectorizer, VECT_PATH)
        logger.info(f"Fitted and saved vectorizer → {VECT_PATH}")
 
    X_text = vectorizer.transform(codes)
 
    # Statistical features
    stat_df = extract_stat_features(codes)
    if not fit and Path(SCALER_PATH).exists():
        scaler = joblib.load(SCALER_PATH)
        logger.info(f"Loaded scaler from {SCALER_PATH}")
    else:
        scaler = StandardScaler()
        scaler.fit(stat_df)
        joblib.dump(scaler, SCALER_PATH)
        logger.info(f"Fitted and saved scaler → {SCALER_PATH}")
 
    X_stats = csr_matrix(scaler.transform(stat_df))
    X = hstack([X_text, X_stats])
    y = df["label"].astype(int).values
    logger.info(f"Feature matrix shape: {X.shape}")
    return X, y, vectorizer, scaler
 
# ─────────────────────────────────────────────
#  Step 3: Training
# ─────────────────────────────────────────────
def train_models(X, y):
    """Train Logistic Regression and Random Forest; save and return both."""
    OUT_DIR.mkdir(exist_ok=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    logger.info(f"Train/test split: {X_train.shape} / {X_test.shape}")
 
    # Logistic Regression
    logreg = LogisticRegression(max_iter=2000, random_state=42)
    logreg.fit(X_train, y_train)
    y_pred_lr = logreg.predict(X_test)
    logger.info("\nLogistic Regression\n" + classification_report(
        y_test, y_pred_lr, target_names=["human", "ai"]
    ))
    logger.info("Confusion matrix:\n" + str(confusion_matrix(y_test, y_pred_lr)))
    joblib.dump(logreg, OUT_DIR / "logreg_model.joblib")
 
    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    logger.info("\nRandom Forest\n" + classification_report(
        y_test, y_pred_rf, target_names=["human", "ai"]
    ))
    logger.info("Confusion matrix:\n" + str(confusion_matrix(y_test, y_pred_rf)))
    joblib.dump(rf, OUT_DIR / "rf_model.joblib")
 
    logger.info(f"Models saved in: {OUT_DIR}/")
    return logreg, rf
 
# ─────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train the AI vs Human code detector.")
    parser.add_argument("--ai_pdf",    default="ai codes.pdf",   help="Path to AI-code PDF")
    parser.add_argument("--human_pdf", default="human code.pdf", help="Path to human-code PDF")
    parser.add_argument("--skip_pdf",  action="store_true",
                        help="Skip PDF extraction and use existing code_dataset.csv")
    args = parser.parse_args()
 
    # Step 1: Dataset
    if args.skip_pdf:
        if not Path(OUT_CSV).exists():
            raise FileNotFoundError(f"{OUT_CSV} not found. Run without --skip_pdf first.")
        df = pd.read_csv(OUT_CSV)
        logger.info(f"Loaded existing dataset: {df.shape}")
    else:
        df = build_dataset(args.ai_pdf, args.human_pdf)
 
    # Step 2: Features
    X, y, vectorizer, scaler = build_features(df, fit=True)
 
    # Step 3: Train
    train_models(X, y)
    logger.info("Training complete. Start the API with:  python app.py")
 
 
if __name__ == "__main__":
    main()
 