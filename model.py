import torch
from torch.nn import Linear, Dropout, Module, Sequential
from torch_geometric.nn import GINEConv, global_mean_pool
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

class HybridGNN(Module):
    def __init__(self, gnn_dim, rdkit_dim, hidden_dim, dropout_rate=0.2, activation='ReLU'):
        super().__init__()
        act_map = {'Swish': torch.nn.SiLU(), 'ReLU': torch.nn.ReLU()}
        act_fn = act_map[activation]
        self.gnn_dim = gnn_dim
        self.rdkit_dim = rdkit_dim

        self.atom_encoder = AtomEncoder(emb_dim=gnn_dim)
        self.bond_encoder = BondEncoder(emb_dim=gnn_dim)

        self.conv1 = GINEConv(Sequential(Linear(gnn_dim, gnn_dim), act_fn, Linear(gnn_dim, gnn_dim)))
        self.conv2 = GINEConv(Sequential(Linear(gnn_dim, gnn_dim), act_fn, Linear(gnn_dim, gnn_dim)))
        self.pool = global_mean_pool

        self.mlp = Sequential(Linear(gnn_dim + rdkit_dim, hidden_dim), act_fn, 
                              Dropout(dropout_rate),
                              Linear(hidden_dim, hidden_dim // 2), act_fn,
                              Dropout(dropout_rate),
                              Linear(hidden_dim // 2, 1))

    def forward(self, data):
        x = self.atom_encoder(data.x)
        edge_attr = self.bond_encoder(data.edge_attr)

        x = self.conv1(x, data.edge_index, edge_attr)
        x = self.conv2(x, data.edge_index, edge_attr)
        x = self.pool(x, data.batch)

        rdkit_feats = getattr(data, 'rdkit_feats', None)
        if rdkit_feats is not None:
            if x.shape[0] != rdkit_feats.shape[0]:
                raise ValueError(f"Shape mismatch: GNN output ({x.shape}) vs rdkit_feats ({rdkit_feats.shape})")
            x = torch.cat([x, rdkit_feats], dim=1)
        else:
            raise ValueError("RDKit features not found in the data object.")

        return self.mlp(x)

def load_model(rdkit_dim: int, path: str = "best_hybridgnn.pt", device: str = "cpu"):
    model = HybridGNN(gnn_dim=512, rdkit_dim=rdkit_dim, hidden_dim=256, dropout_rate=0.29, activation='Swish')
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model
