import os, pathlib, sqlite3, sys, tempfile
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

#  configuration 
DEVICE, RDKIT_DIM, MODEL_PATH, MAX_DISPLAY = "cpu", 6, "best_hybridgnn.pt", 20

#  heavy imports already done above; now Streamlit starts 
@st.cache_resource
def get_model():
    return load_model(rdkit_dim=RDKIT_DIM, path=MODEL_PATH, device=DEVICE)

model = get_model()

# SQLite (cached) — DB stored in /data or /tmp
DB_DIR = pathlib.Path(os.getenv("DB_DIR", "/tmp"))
DB_DIR.mkdir(parents=True, exist_ok=True)

@st.cache_resource
def init_db():
    conn = sqlite3.connect(DB_DIR / "predictions.db", check_same_thread=False)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS predictions(
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               smiles TEXT, prediction REAL, timestamp TEXT)"""
    )
    conn.commit()
    return conn

conn   = init_db()
cursor = conn.cursor()

#  compact info panel 
with st.sidebar.expander("Info & Env", expanded=False):
    st.write(f"Python {sys.version.split()[0]}")
    st.write(f"Temp dir: `{tempfile.gettempdir()}` "
             f"({'writable' if os.access(tempfile.gettempdir(), os.W_OK) else 'read-only'})")
    if "csv_bytes" in st.session_state:
        st.write(f"Last upload: **{len(st.session_state['csv_bytes'])/1024:.1f} KB**")

#  header and instructions (unchanged) 
st.title("HOMO-LUMO Gap Predictor")
st.markdown("""
This app predicts the HOMO-LUMO energy gap for molecules using a trained Graph Neural Network (GNN).

**Instructions:**
- Enter a **single SMILES** string or **comma/newline separated list** in the box below.
- Or **upload a CSV file** containing a single column of SMILES strings.
- **Note**: If you've uploaded a CSV and want to switch to typing SMILES, please click the "X" next to the uploaded file to clear it.
- SMILES format should look like: `O=C(C)Oc1ccccc1C(=O)O` (for aspirin).
- The app will display predictions and molecule images (up to 20 shown at once).
""")

#  uploader (outside the form) 
csv_file = st.file_uploader("CSV with SMILES", type=["csv"])
if csv_file is not None:
    st.session_state["csv_bytes"] = csv_file.getvalue()  # cache raw bytes

#  textarea and button 
smiles_list = []
with st.form("main_form"):
    smiles_text = st.text_area(
        "…or paste SMILES (comma/newline separated)",
        placeholder="CC(=O)Oc1ccccc1C(=O)O",
        height=120,
    )
    run = st.form_submit_button("Run Prediction")

#  decide which input to use 
if run:
    if smiles_text.strip():                      # user typed → override CSV
        smiles_list = [
            s.strip() for s in smiles_text.replace("\n", ",").split(",") if s.strip()
        ]
        st.session_state.pop("csv_bytes", None)  # forget previous upload
        st.success(f"{len(smiles_list)} SMILES parsed from textbox")

    elif "csv_bytes" in st.session_state:        # CSV path
        try:
            df = pd.read_csv(
                StringIO(st.session_state["csv_bytes"].decode("utf-8")),
                comment="#",
            )
            col = df.columns[0] if df.shape[1] == 1 else next(
                (c for c in df.columns if c.lower() == "smiles"), None
            )
            if col is None:
                st.error("CSV needs one column or a 'SMILES' column.")
            else:
                smiles_list = df[col].dropna().astype(str).tolist()
                st.success(f"{len(smiles_list)} SMILES loaded from CSV")
        except Exception as e:
            st.error(f"CSV error: {e}")

    else:
        st.warning("No input provided.")

#  inference & display 
if smiles_list:
    data_list = smiles_to_data(smiles_list, device=DEVICE)
    valid = [(s, d) for s, d in zip(smiles_list, data_list) if d is not None]

    if not valid:
        st.warning("No valid molecules.")
    else:
        vsmi, vdata = zip(*valid)
        preds = []
        for batch in DataLoader(vdata, batch_size=64):
            with torch.no_grad():
                preds.extend(model(batch.to(DEVICE)).view(-1).cpu().numpy().tolist())

        st.subheader(f"Results (first {MAX_DISPLAY})")
        for i, (smi, pred) in enumerate(zip(vsmi, preds)):
            if i >= MAX_DISPLAY:
                st.info("...Only Displaying 20 Compounds")
                break
            mol = Chem.MolFromSmiles(smi)
            if mol:
                st.image(Draw.MolToImage(mol, size=(250, 250)))
            st.write(f"`{smi}` → **{pred:.4f} eV**")

            cursor.execute(
                "INSERT INTO predictions(smiles, prediction, timestamp) VALUES (?,?,?)",
                (smi, float(pred), datetime.utcnow().isoformat()),
            )
        conn.commit()

        st.download_button(
            "Download CSV",
            pd.DataFrame({"SMILES": vsmi, "Gap (eV)": [round(p, 4) for p in preds]})
              .to_csv(index=False).encode(),
            "homolumo_predictions.csv",
            "text/csv",
        )
