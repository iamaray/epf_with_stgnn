import torch
import torch.nn as nn
import torch.optim as optim


class Trainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.mse_criterion = nn.MSELoss()  # For MSE
        self.mae_criterion = nn.L1Loss()  # For MAE
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)  # Example optimizer

    def train(self, data_loader, num_epochs=10):
        for epoch in range(num_epochs):
            for batch in data_loader:
                # Move data to the correct device
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                mse_loss = self.mse_criterion(outputs, targets)
                mae_loss = self.mae_criterion(outputs, targets)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                mse_loss.backward()  # Use MSE for backpropagation
                self.optimizer.step()

                print(f'Epoch [{epoch+1}/{num_epochs}], MSE Loss: {mse_loss.item()}, MAE Loss: {mae_loss.item()}')

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
