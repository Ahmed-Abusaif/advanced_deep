import torch
import torch.nn.functional as F
import numpy as np

class ModelTrainer:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, data, mask):
        self.model.eval()
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            loss = F.nll_loss(out[mask], data.y[mask])
            pred = out[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        return loss.item(), acc

    def train(self, data, epochs=200):
        for epoch in range(epochs):
            train_loss = self.train_epoch(data)
            val_loss, val_acc = self.evaluate(data, data.val_mask)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            if epoch % 20 == 0:
                print(f'Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        return self.train_losses, self.val_losses
