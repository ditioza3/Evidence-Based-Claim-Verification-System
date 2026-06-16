import streamlit as st
import numpy as np
import joblib
import torch
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import torch.nn as nn
import re
import os


os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_META"] = "1"

st.set_page_config(page_title="Claim Verification", layout="wide")
st.title("Evidence based Claim Verification")


CACHE_DIR = Path("./cache")
device = "cpu"

# Load SentenceTransformer
st_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

#----Loading models---------
tfidf = joblib.load(CACHE_DIR / "tfidf_vectorizer.joblib")
label_enc = joblib.load(CACHE_DIR / "label_encoder.joblib")

log_reg  = joblib.load("lr_model.joblib")
xgb_model = joblib.load("xgb_model.joblib")
lgb_model = joblib.load("lgb_model.joblib")
cat_model = joblib.load("cat_model.joblib")


class ImprovedMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.35),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.30),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.25),

            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

INPUT_DIM = 6542
OUTPUT_DIM = len(label_enc.classes_)

mlp_model = ImprovedMLP(INPUT_DIM, OUTPUT_DIM)
mlp_model.load_state_dict(torch.load("mlp_model.pth", map_location=device))
mlp_model.eval()


#------Preprocessing---------
def clean_text(s):
    if s is None: return ""
    s = re.sub(r"<.*?>", " ", str(s))
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[^a-zA-Z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()

def map_verifiable(x):
    s = str(x).lower()
    return 1 if "verif" in s and "not" not in s and not s.startswith("non") else 0

def extract_nums(s):
    return re.findall(r"\d+", str(s))

def num_contradiction(c, e):
    cnums, enums = set(extract_nums(c)), set(extract_nums(e))
    if not cnums or not enums: return -1
    return 1 if cnums.isdisjoint(enums) else 0

def semantic_conflict(c, e, threshold=0.35):
    emb_c = st_model.encode(c, convert_to_tensor=True)
    emb_e = st_model.encode(e, convert_to_tensor=True)
    sim = util.cos_sim(emb_c, emb_e).item()
    if sim < -0.15: return 2
    if sim < threshold: return 1
    return 0

def negation_mismatch(c, e):
    neg_words = {"not", "no", "never", "none", "doesn't", "isnt", "isn't"}
    c_words = c.split()
    e_words = e.split()
    for i, w in enumerate(c_words):
        if w == "not" and i + 1 < len(c_words):
            neg_target = c_words[i+1]
            if neg_target in e_words:
                return 1  

    for i, w in enumerate(e_words):
        if w == "not" and i + 1 < len(e_words):
            neg_target = e_words[i+1]
            if neg_target in c_words:
                return 1 

    return 0

def direct_negation_conflict(c, e):
    e_words = e.split()
    c_words = c.split()
    for i in range(len(e_words) - 1):
        if e_words[i] == "not" and e_words[i+1] in c_words:
            return 1
    return 0


#-----PREDICTION FUNCTION-------

def predict_claim(claim, evidence, verifiable="VERIFIABLE"):

    #---Cleaning------
    c_clean = clean_text(claim)
    e_clean = clean_text(evidence)
    verif_flag = map_verifiable(verifiable)

    #-----Embeddings-----
    c_emb = st_model.encode(c_clean, convert_to_numpy=True)
    e_emb = st_model.encode(e_clean, convert_to_numpy=True)

    #-----Handcrafted features-----
    num_contra = num_contradiction(c_clean, e_clean)
    sem_conf = semantic_conflict(c_clean, e_clean)
    cos_sim = cosine_similarity(c_emb.reshape(1,-1), e_emb.reshape(1,-1))[0,0]
    neg_flag = negation_mismatch(c_clean, e_clean)
    directneg = direct_negation_conflict(c_clean, e_clean)
    handcrafted = np.array([[verif_flag, num_contra, sem_conf, neg_flag, directneg]])

    # Embedding-based features
    emb_features = np.hstack([
        np.abs(c_emb - e_emb).reshape(1, -1),
        (c_emb * e_emb).reshape(1, -1),
        c_emb.reshape(1, -1),
        e_emb.reshape(1, -1),
        np.array([[cos_sim]]),
        handcrafted
    ])

    X_final = hstack([csr_matrix(emb_features), tfidf.transform([c_clean + " " + e_clean])])

    # Model predictions
    lr_pred = label_enc.inverse_transform([log_reg.predict(X_final)[0]])[0]
    xgb_pred = label_enc.inverse_transform([xgb_model.predict(X_final)[0]])[0]
    lgb_pred = label_enc.inverse_transform([lgb_model.predict(X_final)[0]])[0]
    cat_pred = label_enc.inverse_transform([int(cat_model.predict(X_final)[0])])[0]
    with torch.no_grad():
        inp = torch.tensor(X_final.toarray(), dtype=torch.float32)
        mlp_idx = torch.argmax(mlp_model(inp), dim=1).item()
        mlp_pred = label_enc.inverse_transform([mlp_idx])[0]

    #----Post-processing----
    override_label = None

    if sem_conf == 2:
        override_label = "REFUTES"

    elif neg_flag == 1:
        override_label = "REFUTES"

    elif directneg == 1:
        override_label = "REFUTES"

    elif num_contra == 1:
        override_label = "REFUTES"

    if override_label is not None:
        return {m: override_label for m in ["Logistic Regression","XGBoost","LightGBM","CatBoost","MLP"]}

    return {
        "Logistic Regression": lr_pred,
        "XGBoost": xgb_pred,
        "LightGBM": lgb_pred,
        "CatBoost": cat_pred,
        "MLP": mlp_pred
    }


#---------UI-------
st.header("Enter Claim & Evidence")

claim = st.text_area("Claim:", height=100)
evidence = st.text_area("Evidence:", height=100)

verifiable = st.selectbox("Verifiable:", ["VERIFIABLE", "NOT VERIFIABLE"])

if st.button("Predict"):
    if not claim.strip() or not evidence.strip():
        st.error("Please enter both claim and evidence.")
    else:
        with st.spinner("Running models..."):
            results = predict_claim(claim, evidence, verifiable)

        st.subheader("Model Predictions")
        st.write(results)

        supports = list(results.values()).count("SUPPORTS")
        refutes = list(results.values()).count("REFUTES")
        nei = list(results.values()).count("NOT ENOUGH INFO")

        if supports > refutes and supports > nei:
            final = "SUPPORTS"
        elif refutes > supports and refutes > nei:
            final = "REFUTES"
        else:
            final = "NOT ENOUGH INFO"

        st.markdown(f"Final Decision: **{final}**")