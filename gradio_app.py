"""
AI vs Human Code Detector — Gradio UI
======================================
Run: python gradio_app.py
Opens at: http://127.0.0.1:7860
"""
 
import gradio as gr
import joblib
import re
import numpy as np
from scipy.sparse import hstack, csr_matrix
from pathlib import Path
 
# ─────────────────────────────────────────────
#  Load artifacts
# ─────────────────────────────────────────────
vectorizer = joblib.load("tfidf_vectorizer.joblib")
scaler     = joblib.load("scaler.joblib")
logreg     = joblib.load("trained_models/logreg_model.joblib")
 
# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def preprocess_code(code: str) -> str:
    if code is None:
        return ""
    s = str(code)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\n\s*\n+", "\n", s)
    s = s.replace("\t", "    ")
    s = re.sub(r"\b\d{10,}\b", "0", s)
    return s.strip()
 
 
def extract_stat_features(snippets: list[str]) -> np.ndarray:
    rows = []
    for s in snippets:
        s = "" if s is None else str(s)
        lines    = s.splitlines()
        num_lines = max(1, len(lines))
        tokens   = re.findall(r"\w+", s)
        rows.append([
            float(np.mean([len(ln) for ln in lines])) if lines else 0.0,
            sum(1 for ln in lines if ln.strip().startswith("#") or ln.strip().startswith("//")) / num_lines,
            len(re.findall(r"\bimport\b|#include\b", s)),
            len(re.findall(r"\bdef\b|\bclass\b|\bfunction\b", s)),
            len(re.findall(r"[{}()<>;=+\-*/\\\[\]]", s)),
            len(tokens),
            float(np.mean([len(t) for t in tokens])) if tokens else 0.0,
        ])
    return np.array(rows, dtype=float)
 
 
def predict_code(code_text: str) -> str:
    if not code_text or not code_text.strip():
        return "⚠️ Please paste some code."
 
    s           = preprocess_code(code_text)
    X_text      = vectorizer.transform([s])
    stats       = extract_stat_features([s])
    stats_scaled = scaler.transform(stats)
    X_comb      = hstack([X_text, csr_matrix(stats_scaled)])
 
    probs = logreg.predict_proba(X_comb)[0]
    pred  = int(logreg.predict(X_comb)[0])
    label = "AI-Generated 🤖" if pred == 1 else "Human-Written 👤"
 
    return (
        f"Prediction:        {label}\n"
        f"AI Probability:    {probs[1]:.3f}\n"
        f"Human Probability: {probs[0]:.3f}"
    )
 
# ─────────────────────────────────────────────
#  Gradio UI
# ─────────────────────────────────────────────
with gr.Blocks(title="AI Code Detector") as demo:
    gr.Markdown("# 🤖 AI Code Detector\nPaste code below to check if it's AI-generated or human-written.")
 
    code_input = gr.Textbox(
        lines=15,
        placeholder="Paste your code here...",
        label="Code Snippet"
    )
    output = gr.Textbox(label="Result", interactive=False)
    btn    = gr.Button("Predict", variant="primary")
 
    btn.click(fn=predict_code, inputs=code_input, outputs=output)
 
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)