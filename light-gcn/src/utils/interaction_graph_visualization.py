import torch
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import sys 
import os 

def visualize_sampled_interaction_subgraph(edge_index, num_users_to_sample=4, figsize=(12, 8)):
    """
    Samples a subgraph based on a few random users and visualizes it.

    Args:
        edge_index (torch.Tensor): The edge index tensor (shape [2, num_interactions]) 
                                   representing user-item interactions. Should contain positive interactions.
        num_users_to_sample (int): How many users to randomly select for the subgraph.
        figsize (tuple): The size of the matplotlib figure for the plot.
    """
    if not isinstance(edge_index, torch.Tensor) or edge_index.dim() != 2 or edge_index.shape[0] != 2:
        print("Error: Invalid edge_index provided. Must be a [2, num_interactions] Tensor.", file=sys.stderr)
        return

    if edge_index.shape[1] == 0:
        print("Warning: edge_index is empty. Cannot sample or visualize.", file=sys.stderr)
        return
        
    print(f"\n--- Visualizing Subgraph for {num_users_to_sample} Random Users ---")

    # 1. Identify unique users present in the provided edge_index
    try:
        users_present = torch.unique(edge_index[0, :]).numpy()
        n_users_present = len(users_present)
        print(f"Found {n_users_present} unique users in the provided edge_index.")
    except Exception as e:
        print(f"Error identifying unique users from edge_index: {e}", file=sys.stderr)
        return

    # Adjust sampling number if necessary
    actual_users_to_sample = min(num_users_to_sample, n_users_present)
    if actual_users_to_sample == 0:
         print("No users found in edge_index. Cannot sample.", file=sys.stderr)
         return
    if actual_users_to_sample < num_users_to_sample:
        print(f"Warning: Requested {num_users_to_sample}, but only {n_users_present} available. Sampling {actual_users_to_sample}.")

    # 2. Select random users from those present
    sampled_user_indices_indices = random.sample(range(n_users_present), actual_users_to_sample)
    sampled_user_indices = users_present[sampled_user_indices_indices]
    print(f"Sampled User Indices: {sampled_user_indices}")

    # 3. Find interactions involving these users
    try:
        edge_index_np = edge_index.numpy() 
        user_col = edge_index_np[0, :]
        mask = np.isin(user_col, sampled_user_indices)
        sampled_edges_np = edge_index_np[:, mask]
    except Exception as e:
        print(f"Error filtering edges: {e}", file=sys.stderr)
        return

    # 4. Identify unique items connected to these users
    sampled_item_indices = np.unique(sampled_edges_np[1, :])
    print(f"Found {len(sampled_item_indices)} items connected to these users.")

    # 5. Create the Subgraph in NetworkX
    G_sampled = nx.Graph()

    sampled_user_nodes = [f'U{i}' for i in sampled_user_indices]
    sampled_item_nodes = [f'I{i}' for i in sampled_item_indices]

    G_sampled.add_nodes_from(sampled_user_nodes, bipartite=0)
    G_sampled.add_nodes_from(sampled_item_nodes, bipartite=1)

    for i in range(sampled_edges_np.shape[1]):
        u_idx = sampled_edges_np[0, i]
        i_idx = sampled_edges_np[1, i]
        G_sampled.add_edge(f'U{u_idx}', f'I{i_idx}')

    print(f"Sampled subgraph created with {G_sampled.number_of_nodes()} nodes and {G_sampled.number_of_edges()} edges.")
    if G_sampled.number_of_edges() == 0 and len(sampled_user_nodes) > 0:
         print("Warning: Sampled users have no corresponding edges in the provided edge_index.")

    # 6. Visualize the Subgraph
    plt.figure(figsize=figsize) 
    try:
        # Ensure bipartite_layout has nodes from both sets if edges exist
        if G_sampled.number_of_edges() > 0 or len(sampled_item_nodes) > 0:
             pos = nx.bipartite_layout(G_sampled, sampled_user_nodes) 
        else: # Handle cases with isolated users
             pos = nx.spring_layout(G_sampled) # Fallback layout

        nx.draw_networkx_nodes(G_sampled, pos, nodelist=sampled_user_nodes, node_color='skyblue', node_size=500)
        nx.draw_networkx_nodes(G_sampled, pos, nodelist=sampled_item_nodes, node_color='lightgreen', node_size=500)
        nx.draw_networkx_edges(G_sampled, pos, edge_color='gray', alpha=0.6)
        nx.draw_networkx_labels(G_sampled, pos, font_size=8)

        plt.title(f"Sampled Interaction Subgraph ({actual_users_to_sample} Users)") 
        plt.box(False)
        plt.show()
    except Exception as e:
         print(f"Error during graph visualization: {e}", file=sys.stderr)
