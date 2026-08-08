import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGCN(nn.Module):
    """
    Original GCN branch preserved for reference.
    Subject to over-smoothing on small, dense functional connectivity graphs.
    """
    def __init__(self, num_nodes, in_features, hidden_dim, out_dim):
        super().__init__()
        self.num_nodes = num_nodes
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x, adj):
        B, N, _ = adj.shape
        I = torch.eye(N, device=adj.device).unsqueeze(0).expand(B, -1, -1)
        A_hat = adj + I
        row_sum = torch.sum(A_hat, dim=-1)
        d_inv_sqrt = torch.pow(torch.clamp(row_sum, min=1e-6), -0.5)
        D_hat = torch.diag_embed(d_inv_sqrt)
        norm_adj = torch.bmm(torch.bmm(D_hat, A_hat), D_hat)
        
        out = self.fc1(x)
        out = torch.bmm(norm_adj, out)
        out = F.relu(out)
        
        out = self.fc2(out)
        out = torch.bmm(norm_adj, out)
        out = F.relu(out)
        
        out = torch.mean(out, dim=1) 
        return out

class GNN(nn.Module):
    """
    Consolidated Functional Connectivity Projection MLP.
    Extracts the unique upper triangular elements of the symmetric PLI connectivity
    matrix (excluding the diagonal) and projects them through a regularized MLP.
    
    This preserves the specific pairwise connectivity values (edges) without 
    suffering from GNN over-smoothing on small dense graphs.
    """
    def __init__(self, num_nodes=16, embed_dim=64):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_nodes * (num_nodes - 1) // 2
        
        self.fc = nn.Sequential(
            nn.Linear(self.num_edges, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Precompute upper triangular indices to extract edges
        triu_indices = torch.triu_indices(row=num_nodes, col=num_nodes, offset=1)
        self.register_buffer('triu_row', triu_indices[0])
        self.register_buffer('triu_col', triu_indices[1])
        
    def forward(self, adj):
        # adj shape: (B, N, N)
        # Extract unique edge values: shape (B, num_edges)
        x = adj[:, self.triu_row, self.triu_col]
        return self.fc(x)
