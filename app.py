import streamlit as st
import pandas as pd
import torch
import sqlite3
from datetime import datetime
from rdkit import Chem
from rdkit.Chem import Draw
import os, pathlib, sys
import tempfile
from io import StringIO, BytesIO
from model import load_model
from utils import smiles_to_data
from torch_geometric.loader import DataLoader

# Config 
DEVICE = "cpu"
RDKIT_DIM = 6
MODEL_PATH = "best_hybridgnn.pt"
MAX_DISPLAY = 10

# Debug sidebar
with st.sidebar:
    st.title("Debug Tools")
    if st.button("Show Environment Info"):
        st.write("### System Info")
        st.write(f"Python version: {sys.version}")
        st.write(f"Current working directory: {os.getcwd()}")
        st.write(f"Temp directory: {tempfile.gettempdir()}")
        st.write(f"Temp dir exists: {os.path.exists(tempfile.gettempdir())}")
        st.write(f"Temp dir writable: {os.access(tempfile.gettempdir(), os.W_OK)}")
        st.write(f"Current user: {os.getenv('USER', 'unknown')}")
        
        try:
            st.write("### Directory Contents")
            st.write(f"Files in current directory: {os.listdir('.')}")
            st.write(f"Files in /tmp: {os.listdir('/tmp')}")
        except Exception as e:
            st.error(f"Error listing directories: {e}")
            
        st.write("### Environment Variables")
        for key, value in os.environ.items():
            if not key.startswith(('AWS', 'SECRET')):  # Skip sensitive vars
                st.write(f"{key}: {value}")

# Load Model 
@st.cache_resource
def load_cached_model():
    try:
        return load_model(rdkit_dim=RDKIT_DIM, path=MODEL_PATH, device=DEVICE)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_cached_model()

# SQLite Setup 
DB_DIR = os.getenv("DB_DIR", "/tmp")
pathlib.Path(DB_DIR).mkdir(parents=True, exist_ok=True)

@st.cache_resource
def init_db():
    try:
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
    except Exception as e:
        st.error(f"Database initialization error: {e}")
        return None

conn = init_db()
if conn:
    cursor = conn.cursor()

# Streamlit UI 
st.title("HOMO-LUMO Gap Predictor")
st.markdown("""
This app predicts the HOMO-LUMO energy gap for molecules using a trained Graph Neural Network (GNN).

**Instructions:**
- Enter a **single SMILES** string or **comma-separated list** in the box below.
- Or **upload a CSV file** containing a single column of SMILES strings.
- **Note**: If you've uploaded a CSV and want to switch to typing SMILES, please click the "X" next to the uploaded file to clear it.
- SMILES format should look like: `CC(=O)Oc1ccccc1C(=O)O` (for aspirin).
- The app will display predictions and molecule images (up to 10 shown at once).
""")

# File handling with caching
@st.cache_data
def read_csv_file(file_content):
    """Cache the file reading operation"""
    try:
        # Try to read as string first
        if isinstance(file_content, str):
            df = pd.read_csv(StringIO(file_content), comment="#")
        else:
            # If it's bytes, decode it
            df = pd.read_csv(StringIO(file_content.decode('utf-8')), comment="#")
        return df, None
    except Exception as e:
        return None, str(e)

# Debug container for file upload messages
file_debug = st.container()

# File uploader outside the form
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    with file_debug:
        st.write(f"File name: {uploaded_file.name}")
        st.write(f"File type: {uploaded_file.type}")
        st.write(f"File size: {uploaded_file.size} bytes")

with st.form("input_form"):
    smiles_input = st.text_area(
        "Enter SMILES string(s)", 
        placeholder="C1=CC=CC=C1, CC(=O)Oc1ccccc1C(=O)O", 
        height=120
        )
    run_button = st.form_submit_button("Submit")

smiles_list = []

# Process after the button press
if run_button:
    #  CSV path 
    if uploaded_file is not None:
        with file_debug:
            st.write("### Processing CSV file")
            try:
                # Save file temporarily for debugging
                temp_file = os.path.join(tempfile.gettempdir(), uploaded_file.name)
                with open(temp_file, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                st.write(f"Saved temporary file at: {temp_file}")
                st.write(f"File exists: {os.path.exists(temp_file)}")
                st.write(f"File size on disk: {os.path.getsize(temp_file)} bytes")
                
                # Read file content
                file_content = uploaded_file.getvalue()
                st.write(f"Read {len(file_content)} bytes from file")
                
                # Try to decode first few bytes
                preview = file_content[:100] if len(file_content) > 100 else file_content
                try:
                    decoded_preview = preview.decode('utf-8')
                    st.write(f"File preview (decoded): {decoded_preview}")
                except:
                    st.write(f"File preview (hex): {preview.hex()}")
                
                # Use cached reading function
                df, error = read_csv_file(file_content)
                
                if error:
                    st.error(f"CSV reading error: {error}")
                elif df is not None:
                    st.write(f"CSV loaded with {df.shape[0]} rows and {df.shape[1]} columns")
                    st.write("CSV columns:", df.columns.tolist())
                    st.write("First few rows:", df.head())

                    # choose the SMILES column
                    if df.shape[1] == 1:
                        smiles_col = df.iloc[:, 0]
                        st.write("Using the only column for SMILES")
                    elif "smiles" in [c.lower() for c in df.columns]:
                        col_name = [c for c in df.columns if c.lower() == "smiles"][0]
                        smiles_col = df[col_name]
                        st.write(f"Using column '{col_name}' for SMILES")
                    else:
                        st.error(f"CSV must have a single column or a column named 'SMILES'. Found columns: {', '.join(df.columns)}")
                        st.write("Using first column as fallback")
                        smiles_col = df.iloc[:, 0]

                    smiles_list = smiles_col.dropna().astype(str).tolist()
                    st.success(f"{len(smiles_list)} SMILES loaded from CSV")
                    if smiles_list:
                        st.write("First few SMILES:", smiles_list[:5])
                else:
                    st.error("Failed to process CSV: DataFrame is None")
            except Exception as e:
                st.error(f"Critical error processing CSV: {str(e)}")
                st.exception(e)  # This shows the full traceback

    # Textarea path 
    elif smiles_input.strip():
        raw_input = smiles_input.replace("\n", ",")
        smiles_list = [s.strip() for s in raw_input.split(",") if s.strip()]
        st.success(f"{len(smiles_list)} SMILES parsed from text")
        if smiles_list:
            st.write("First few SMILES:", smiles_list[:5])

# Run Inference 
if smiles_list:
    with st.spinner("Processing molecules..."):
        try:
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

                    # Log to SQLite if connection exists
                    if conn:
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
        except Exception as e:
            st.error(f"Error during inference: {str(e)}")
            st.exception(e)  # This shows the full traceback