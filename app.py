import os
import pathlib
import sqlite3
from datetime import datetime
from io import StringIO

import pandas as pd
import streamlit as st
import torch
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.loader import DataLoader

from model import load_model
from utils import smiles_to_data

# ───────────────────────── Configuration ─────────────────────────
DEVICE       = "cpu"
RDKIT_DIM    = 6
MODEL_PATH   = "best_hybridgnn.pt"
MAX_DISPLAY  = 10

# ─────────────────────── Cached model & database ─────────────────
@st.cache_resource
def get_model():
    return load_model(rdkit_dim=RDKIT_DIM, path=MODEL_PATH, device=DEVICE)

model = get_model()

DB_DIR = pathlib.Path(os.getenv("DB_DIR", "/tmp"))
DB_DIR.mkdir(parents=True, exist_ok=True)

@st.cache_resource
def init_db():
    conn = sqlite3.connect(DB_DIR / "predictions.db", check_same_thread=False)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            smiles TEXT,
            prediction REAL,
            timestamp TEXT
        )
        """
    )
    conn.commit()
    return conn

conn   = init_db()
cursor = conn.cursor()

#  UI header 
st.title("HOMO-LUMO Gap Predictor")
st.markdown(
    """
    Paste SMILES **or** upload a one-column CSV, then click **Run Prediction**.
    The app draws each molecule and shows the predicted HOMO-LUMO gap (eV).
    """
)

#  Input widgets 
csv_file = st.file_uploader("Upload CSV (one SMILES column)", type=["csv"])
if csv_file is not None:
    st.session_state["uploaded_csv"] = csv_file  # persist across reruns

smiles_list = []

with st.form("smiles_or_csv"):
    smiles_text = st.text_area(
        "…or paste SMILES (comma or newline separated)",
        placeholder="CC(=O)Oc1ccccc1C(=O)O",
        height=120,
    )
    run = st.form_submit_button("Run Prediction")

#  Parse input after button 
if run:
    csv_obj = st.session_state.get("uploaded_csv", None)

    #  CSV branch 
    if csv_obj is not None:
        try:
            csv_obj.seek(0)
            df = pd.read_csv(StringIO(csv_obj.getvalue().decode("utf-8")), comment="#")

            if df.shape[1] == 1:
                smiles_col = df.iloc[:, 0]
            elif "smiles" in [c.lower() for c in df.columns]:
                smiles_col = df[
                    [c for c in df.columns if c.lower() == "smiles"][0]
                ]
            else:
                st.error(
                    "CSV must have one column **or** a column named 'SMILES'"
                    f"Found: {', '.join(df.columns)}"
                )
                smiles_col = None

            if smiles_col is not None:
                smiles_list = smiles_col.dropna().astype(str).tolist()
                st.success(f"{len(smiles_list)} SMILES loaded from CSV")
        except Exception as e:
            st.error(f"CSV read error: {e}")

    #  Textarea branch 
    elif smiles_text.strip():
        raw = smiles_text.replace("\n", ",")
        smiles_list = [s.strip() for s in raw.split(",") if s.strip()]
        st.success(f"{len(smiles_list)} SMILES parsed from textbox")
    else:
        st.warning("Paste SMILES or upload a CSV before pressing **Run**")

#  Inference 
if smiles_list:
    with st.spinner("Running model…"):
        data_list = smiles_to_data(smiles_list, device=DEVICE)

    valid_pairs = [
        (smi, data)
        for smi, data in zip(smiles_list, data_list)
        if data is not None
    ]

    if not valid_pairs:
        st.warning("No valid molecules found")
    else:
        valid_smiles, valid_data = zip(*valid_pairs)
        loader = DataLoader(valid_data, batch_size=64)
        preds  = []

        for batch in loader:
            batch = batch.to(DEVICE)
            with torch.no_grad():
                preds.extend(model(batch).view(-1).cpu().numpy().tolist())

        #  Display results 
        st.subheader(f"Predictions (showing up to {MAX_DISPLAY})")
        for i, (smi, pred) in enumerate(zip(valid_smiles, preds)):
            if i >= MAX_DISPLAY:
                st.info(f"…only first {MAX_DISPLAY} molecules shown")
                break
            mol = Chem.MolFromSmiles(smi)
            if mol:
                st.image(Draw.MolToImage(mol, size=(250, 250)))
            st.write(f"**SMILES:** `{smi}`")
            st.write(f"**Predicted Gap:** `{pred:.4f} eV`")

            cursor.execute(
                "INSERT INTO predictions (smiles, prediction, timestamp) VALUES (?, ?, ?)",
                (smi, float(pred), datetime.now().isoformat())
            )
        conn.commit()

        #  Download results 
        res_df = pd.DataFrame(
            {
                "SMILES": valid_smiles,
                "Predicted HOMO-LUMO Gap (eV)": [round(p, 4) for p in preds],
            }
        )
        st.download_button(
            "Download results as CSV",
            res_df.to_csv(index=False).encode("utf-8"),
            "homolumo_predictions.csv",
            "text/csv",
        )
