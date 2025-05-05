# HOMO–LUMO Gap Predictor

This web app uses a trained Graph Neural Network (GNN) to predict HOMO–LUMO energy gaps from molecular SMILES strings. Built with [Streamlit](https://streamlit.io), it enables fast single or batch predictions with visualization.

### Live App

[Click here to launch the app](https://www.willfillinoncedeployed.com)  


---

## Features

- Predict HOMO–LUMO gap for one or many molecules
- Accepts comma-separated SMILES or CSV uploads
- RDKit rendering of molecule structures
- Downloadable CSV of predictions
- Powered by a trained hybrid GNN model with RDKit descriptors

---

## Usage

1. **Input Options**:
   - Type one or more SMILES strings separated by commas
   - OR upload a `.csv` file with a single column of SMILES

2. **Example SMILES**: CC(=O)Oc1ccccc1C(=O)O, C1=CC=CC=C1

3. **CSV Format**:
- One column
- No header 
- Each row contains a SMILES string

4. **Output**:
- Predictions displayed in-browser (up to 10 molecules shown)
- Full results available for download as CSV

---

## Project Structure

streamlit-app/
│
├── app.py # Main Streamlit app
├── model.py # Hybrid GNN architecture and model loader
├── utils.py # RDKit and SMILES processing
├── requirements.txt # Python dependencies
└── predictions.db # SQLite log of predictions 

---

## Requirements

To run locally:
```
pip install -r requirements.txt
streamlit run app.py

```


## Model Info

The app uses a trained hybrid GNN model combining:

* AtomEncoder and BondEncoder from OGB
* GINEConv layers from PyTorch Geometric
* Global mean pooling
* RDKit-based physicochemical descriptors

Trained on the [OGB PCQM4Mv2 dataset](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/), optimized using Optuna


## Author

Developed by [Matthew Graham](https://github.com/MooseML)
For inquiries, collaborations, or ideas, feel free to reach out!







