import torch
import torch.nn as nn
import torch.nn.functional as F
import sys 

try:
    from torch_geometric.nn import GCNConv
    from torch_geometric.utils import degree # Use degree for manual normalization
    PYG_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: PyTorch Geometric import failed: {e}", file=sys.stderr)
    print("Please ensure torch_geometric is installed correctly.", file=sys.stderr)
    PYG_AVAILABLE = False
    class GCNConv: pass 
    def degree(*args, **kwargs): print("Dummy degree called"); return torch.ones(1) 

class LightGCN(nn.Module): 
    def __init__(self, n_users, n_items, embed_dim, n_layers=3):
        super(LightGCN, self).__init__()

        if not PYG_AVAILABLE:
             raise ImportError("torch_geometric ('GCNConv', 'degree') could not be imported.")

        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        print(f"Initializing LightGCN model with {self.n_layers} layers...") # Updated print statement

        self.user_embedding = nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.embed_dim)
        self.item_embedding = nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.embed_dim)
        
        # GCNConv for propagation, configured for LightGCN style
        self.gcn_conv = GCNConv(in_channels=self.embed_dim, 
                                out_channels=self.embed_dim, 
                                normalize=False, # Manual normalization applied in get_embeddings
                                add_self_loops=False, 
                                bias=False) 
        
        # Initialize weights
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        print("Initialized user and item embedding layers.")

    def get_embeddings(self, edge_index):
        """Computes user and item embeddings using LightGCN propagation."""
        user_embed_0 = self.user_embedding.weight
        item_embed_0 = self.item_embedding.weight
        x0 = torch.cat([user_embed_0, item_embed_0], dim=0)
        
        all_layer_embeddings = [x0] 
        
        # Prepare full bipartite graph edge index and compute normalization weights
        row, col = edge_index
        num_nodes = self.n_users + self.n_items
        adjusted_col = col + self.n_users # Offset item indices for full graph
        
        # Edges for user->item and item->user
        full_edge_index = torch.cat([
            torch.stack([row, adjusted_col], dim=0), 
            torch.stack([adjusted_col, row], dim=0) 
        ], dim=1)
        
        # --- Manually Calculate Symmetric Normalization Weights (D^-0.5 * A * D^-0.5) ---
        deg = degree(full_edge_index[0], num_nodes=num_nodes, dtype=x0.dtype).clamp(min=1)
        deg_inv_sqrt = deg.pow(-0.5)
        norm_edge_weight = deg_inv_sqrt[full_edge_index[0]] * deg_inv_sqrt[full_edge_index[1]]
        # --- End Manual Normalization ---

        current_embeddings = x0
        for layer in range(self.n_layers):
            next_embeddings = self.gcn_conv.propagate(
                full_edge_index, 
                x=current_embeddings, 
                edge_weight=norm_edge_weight, 
                size=(num_nodes, num_nodes) # Specify shape for propagation
            )
            all_layer_embeddings.append(next_embeddings)
            current_embeddings = next_embeddings 
            
        # Aggregate embeddings from all layers (LightGCN style: mean aggregation)
        final_embeddings_stack = torch.stack(all_layer_embeddings, dim=0)
        final_embeddings = torch.mean(final_embeddings_stack, dim=0) 
       
        # Split aggregated embeddings back into users and items
        final_user_embed, final_item_embed = torch.split(
            final_embeddings, [self.n_users, self.n_items], dim=0
        )
        
        return final_user_embed, final_item_embed

    def forward(self, edge_index, users_idx=None, pos_items_idx=None, neg_items_idx=None):
        """
        Forward pass. 
        If user/item indices are provided, computes scores for BPR loss.
        Otherwise, returns all final embeddings.
        """
        final_user_embed, final_item_embed = self.get_embeddings(edge_index)
        
        # Mode 1: Return all embeddings (for evaluation/inference)
        if users_idx is None or pos_items_idx is None or neg_items_idx is None:
            return final_user_embed, final_item_embed
            
        # Mode 2: Compute scores for BPR loss (for training)
        # Select embeddings for the batch
        users_emb_final = final_user_embed[users_idx]
        pos_items_emb_final = final_item_embed[pos_items_idx]
        neg_items_emb_final = final_item_embed[neg_items_idx]
        
        # Calculate scores
        pos_scores = torch.sum(users_emb_final * pos_items_emb_final, dim=1)
        neg_scores = torch.sum(users_emb_final * neg_items_emb_final, dim=1)
        
        # Get initial embeddings for regularization (needed for BPR loss)
        user_emb_initial = self.user_embedding(users_idx)
        item_emb_pos_initial = self.item_embedding(pos_items_idx)
        item_emb_neg_initial = self.item_embedding(neg_items_idx)
        
        return pos_scores, neg_scores, user_emb_initial, item_emb_pos_initial, item_emb_neg_initial