import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import Planetoid

def load_cora_dataset():
    dataset = Planetoid(root='/Cora', name='Cora')
    data = dataset[0]
    return data

def print_dataset_info(data):
    print(f"Number of nodes: {data.x.shape[0]}")
    print(f"Number of features: {data.x.shape[1]}")
    print(f"Number of classes: {len(data.y.unique())}")
    print(f"Number of edges: {data.edge_index.shape[1]}")
    print(f"Training nodes: {data.train_mask.sum().item()}")
    print(f"Validation nodes: {data.val_mask.sum().item()}")
    print(f"Test nodes: {data.test_mask.sum().item()}")

def visualize_graph(data):
    G = to_networkx(data)
    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])
    nx.draw_spring(G, node_size=50)
    plt.show()
