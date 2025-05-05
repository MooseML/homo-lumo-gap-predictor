import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch_geometric.data import Data
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from ogb.lsc import PCQM4Mv2Evaluator
from ogb.utils import smiles2graph
from torch_geometric.loader import DataLoader

def compute_rdkit_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    return [
        Descriptors.MolWt(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.RingCount(mol)
    ]

def smiles_to_data(smiles_list, device="cpu"):
    graph_list = []
    rdkit_list = []

    for smi in smiles_list:
        try:
            graph = smiles2graph(smi)
            rdkit_feats = compute_rdkit_features(smi)

            data = Data(
                x=torch.tensor(graph['node_feat'], dtype=torch.long),
                edge_index=torch.tensor(graph['edge_index'], dtype=torch.long),
                edge_attr=torch.tensor(graph['edge_feat'], dtype=torch.long),
                rdkit_feats=torch.tensor(rdkit_feats, dtype=torch.float32).unsqueeze(0),
                num_nodes=graph['num_nodes']
            )
            graph_list.append(data)
        except Exception as e:
            print(f"Error with SMILES '{smi}': {e}")
            continue

    return graph_list

