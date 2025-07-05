import torch
import matplotlib.pyplot as plt
from utils import load_cora_dataset, print_dataset_info, visualize_graph
from models import GCN, GraphSAGE
from trainer import ModelTrainer

def run_experiment(model_class, hidden_units, learning_rate, dropout):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = load_cora_dataset()
    data = data.to(device)

    model = model_class(
        num_features=data.x.shape[1],
        hidden_units=hidden_units,
        num_classes=len(data.y.unique()),
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    trainer = ModelTrainer(model, optimizer, device)
    
    train_losses, val_losses = trainer.train(data)
    _, test_acc = trainer.evaluate(data, data.test_mask)
    
    return train_losses, val_losses, test_acc

def main():
    # Dataset exploration
    data = load_cora_dataset()
    print_dataset_info(data)
    visualize_graph(data)

    # Hyperparameter configurations
    configs = [
        {'hidden_units': h, 'lr': lr, 'dropout': d}
        for h in [16, 32, 64]
        for lr in [0.01, 0.001]
        for d in [0.0, 0.5]
    ]

    # Run experiments with GCN
    gcn_results = []
    for config in configs:
        print(f"\nRunning GCN with config: {config}")
        train_losses, val_losses, test_acc = run_experiment(
            GCN, config['hidden_units'], config['lr'], config['dropout']
        )
        gcn_results.append((config, test_acc))

    # Run experiment with GraphSAGE
    print("\nRunning GraphSAGE with best GCN config")
    best_config = max(gcn_results, key=lambda x: x[1])[0]
    sage_train_losses, sage_val_losses, sage_test_acc = run_experiment(
        GraphSAGE, best_config['hidden_units'], best_config['lr'], best_config['dropout']
    )

    print("\nResults:")
    print(f"Best GCN config: {best_config}")
    print(f"Best GCN test accuracy: {max(gcn_results, key=lambda x: x[1])[1]:.4f}")
    print(f"GraphSAGE test accuracy: {sage_test_acc:.4f}")

if __name__ == "__main__":
    main()
