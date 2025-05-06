---
title: 🧪 HOMO‑LUMO Gap Predictor
emoji: 🧬
colorFrom: indigo
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# HOMO‑LUMO Gap Predictor — Streamlit + Hybrid GNN

> **Live demo:** [huggingface.co/spaces/MooseML/homo-lumo-gap-predictor](https://huggingface.co/spaces/MooseML/homo-lumo-gap-predictor) &nbsp;•&nbsp;  
> **Code:** <https://github.com/MooseML/homo-lumo-gap-predictor>

This web app predicts HOMO–LUMO energy gaps from molecular **SMILES** using a trained hybrid Graph Neural Network (PyTorch Geometric and RDKit descriptors).  
It runs on Hugging Face Spaces via Docker and on any local machine with Docker or a Python ≥ 3.10 environment.

---

## Features

* Predict gaps for **single or batch** inputs (comma / newline SMILES or CSV upload)
* Shows **up to 20 molecules** per run with RDKit 2‑D depictions
* Download full predictions as CSV
* Logs all predictions to a lightweight SQLite DB (`/data/predictions.db`)
* Containerised environment identical to the public Space

---

## Quick start

### 1  Use the hosted Space

1. Open the [live URL](https://huggingface.co/spaces/MooseML/homo-lumo-gap-predictor).  
2. Paste SMILES *or* upload a CSV (1 column, no header).  
3. Click **Run Prediction** → results and structures appear; a CSV is downloadable.

> **Heads‑up:** on the free HF tier large files (> ~5 MB) can take 10–30 s to upload because of proxy buffering.  
> Local Docker runs are instant, see below:

### 2  Run locally with Docker (mirrors the Space)

```bash
git clone https://github.com/MooseML/homo-lumo-gap-predictor.git
cd homo-lumo-gap-predictor
docker build -t homolumo .
docker run -p 7860:7860 homolumo
# open http://localhost:7860
````

### 3  Run locally with Python (no Docker)

```bash
git clone https://github.com/MooseML/homo-lumo-gap-predictor.git
cd homo-lumo-gap-predictor
pip install -r requirements.txt
streamlit run app.py 
# app on http://localhost:8501
```

---

## Input guidelines

| Format       | Example                                        |
| ------------ | ---------------------------------------------- |
| **Textarea** | `O=C(C)Oc1ccccc1C(=O)O, C1=CC=CC=C1`           |
| **CSV**      | One column, no header:<br>`CCO`<br>`Cc1ccccc1` |

Invalid or exotic SMILES are skipped and listed in the terminal log (RDKit warnings)

---

## Project files

```
.
├── app.py          – Streamlit front‑end
├── model.py        – Hybrid GNN loader (PyTorch Geometric)
├── utils.py        – RDKit helpers & SMILES→graph
├── Dockerfile      – identical to the Hugging Face Space
└── requirements.txt
```

The Docker image creates `/data` (writable, 775) for the persistent SQLite DB when a volume is attached.

---

## Model in brief

* **Architecture:** AtomEncoder and BondEncoder → GINEConv layers → global mean pooling → dense head
* **Descriptors:** six RDKit physico‑chemical features per molecule
* **Training set:** [OGB PCQM4Mv2](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/)
* **Optimiser / search:** Optuna hyperparameter sweep

---

## Roadmap

* Stream chunked CSV parsing to improve upload speed on the public Space
* Toggle between 2‑D and 3‑D (3Dmol.js) molecule renderings
* Serve the model weights from the HF Hub instead of bundling in the image

---

## Author

**Matthew Graham** — [@MooseML](https://github.com/MooseML)
Feel free to open issues or contact me for collaborations.

```
```
