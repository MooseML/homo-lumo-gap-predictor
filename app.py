import streamlit as st
import pandas as pd
import torch
import sqlite3
from datetime import datetime
from rdkit import Chem
from rdkit.Chem import Draw
import os, pathlib
from io import StringIO 
from model import load_model
from utils import smiles_to_data
from torch_geometric.loader import DataLoader

# Config 
DEVICE = "cpu"
RDKIT_DIM = 6
MODEL_PATH = "best_hybridgnn.pt"
MAX_DISPLAY = 10

# Load Model 
model = load_model(rdkit_dim=RDKIT_DIM, path=MODEL_PATH, device=DEVICE)

# SQLite Setup 
DB_DIR = os.getenv("DB_DIR", "/tmp")      # /data if you add a volume later
pathlib.Path(DB_DIR).mkdir(parents=True, exist_ok=True)

@st.cache_resource
def init_db():
    db_file = os.path.join(DB_DIR, "predictions.db")
    conn = sqlite3.connect(db_file, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            smiles TEXT,
            prediction REAL,
            timestamp TEXT
        )
    """)
    conn.commit()
    return conn

conn = init_db()
cursor = conn.cursor()

# Streamlit UI 
st.title("HOMO-LUMO Gap Predictor")
st.markdown("""
This app predicts the HOMO-LUMO energy gap for molecules using a trained Graph Neural Network (GNN).

**Instructions:**
- Enter a **single SMILES** string or **comma-separated list** in the box below.
- Or **upload a CSV file** containing a single column of SMILES strings.
- **Note**: If you've uploaded a CSV and want to switch to typing SMILES, please click the “X” next to the uploaded file to clear it.
- SMILES format should look like: `CC(=O)Oc1ccccc1C(=O)O` (for aspirin).
- The app will display predictions and molecule images (up to 10 shown at once).
""")




#  Single input form 
smiles_list = []

with st.form("smiles_or_csv"):
    smiles_text = st.text_area(
        "SMILES (comma or line-separated)",
        placeholder="C1=CC=CC=C1\nCC(=O)Oc1ccccc1C(=O)O",
        height=120,
    )
    csv_file = st.file_uploader(
        "…or upload a one-column CSV",
        type=["csv"],
    )
    run = st.form_submit_button("Run Prediction")

if run:
    if csv_file is not None:
        try:
            csv_file.seek(0)
            df = pd.read_csv(StringIO(csv_file.getvalue().decode("utf‑8")), comment="#")

            # pick SMILES column
            if df.shape[1] == 1:
                smiles_col = df.iloc[:, 0]
            elif "smiles" in [c.lower() for c in df.columns]:
                smiles_col = df[[c for c in df.columns if c.lower() == "smiles"][0]]
            else:
                st.error(
                    "CSV must have a single column **or** a column named 'SMILES'. "
                    f"Found columns: {', '.join(df.columns)}"
                )
                smiles_col = None

            if smiles_col is not None:
                smiles_list = smiles_col.dropna().astype(str).tolist()
                st.success(f"{len(smiles_list)} SMILES loaded from CSV.")
        except Exception as e:
            st.error(f"CSV read error: {e}")

    elif smiles_text.strip():
        raw = smiles_text.replace("\n", ",")
        smiles_list = [s.strip() for s in raw.split(",") if s.strip()]
        st.success(f"{len(smiles_list)} SMILES parsed from textbox.")
    else:
        st.warning("Please paste SMILES or upload a CSV before pressing *Run*.")


# Run Inference 
if smiles_list:
    with st.spinner("Processing molecules..."):
        data_list = smiles_to_data(smiles_list, device=DEVICE)

        # Filter only valid molecules and keep aligned SMILES
        valid_pairs = [(smi, data) for smi, data in zip(smiles_list, data_list) if data is not None]

        if not valid_pairs:
            st.warning("No valid molecules found")
        else:
            valid_smiles, valid_data = zip(*valid_pairs)
            loader = DataLoader(valid_data, batch_size=64)
            predictions = []

            for batch in loader:
                batch = batch.to(DEVICE)
                with torch.no_grad():
                    pred = model(batch).view(-1).cpu().numpy()
                    predictions.extend(pred.tolist())

            # Display Results 
            st.subheader(f"Predictions (showing up to {MAX_DISPLAY} molecules):")

            for i, (smi, pred) in enumerate(zip(valid_smiles, predictions)):
                if i >= MAX_DISPLAY:
                    st.info(f"...only showing the first {MAX_DISPLAY} molecules")
                    break

                mol = Chem.MolFromSmiles(smi)
                if mol:
                    st.image(Draw.MolToImage(mol, size=(250, 250)))
                st.write(f"**SMILES**: `{smi}`")
                st.write(f"**Predicted HOMO-LUMO Gap**: `{pred:.4f} eV`")

                # Log to SQLite 
                cursor.execute("INSERT INTO predictions (smiles, prediction, timestamp) VALUES (?, ?, ?)",
                               (smi, pred, str(datetime.now())))
                conn.commit()

            # Download Results 
            result_df = pd.DataFrame({"SMILES": valid_smiles, 
                                      "Predicted HOMO-LUMO Gap (eV)": [round(p, 4) for p in predictions]})

            st.download_button(label="Download Predictions as CSV", 
                               data=result_df.to_csv(index=False).encode('utf-8'),
                               file_name="homolumo_predictions.csv",
                               mime="text/csv")
