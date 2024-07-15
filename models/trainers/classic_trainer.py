import torch
import numpy as np
import torch.optim as optim

from trainers.error_metrics import *


class Trainer:
    def __init__(
            self,
            model,
            epochs=60,
            lr=1e-3,
            weight_decay=1e-4,
            grad_clip=5,
            criterion=masked_mae):

        self.model = model
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.epochs = epochs
        self.criterion = criterion
        self.losses = np.array([])
        self.grad_clip = grad_clip

    def train(self, train_loader):
        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for data in train_loader:
                data = data.to('cuda')
                self.optimizer.zero_grad()
                outputs = self.model(data)
                if self.model.use_graph_conv and not self.model.use_temp_conv:
                    outputs = outputs.permute(0, 2, 1)

                loss = self.criterion(outputs, data.y, 0.0)
                # loss = self.criterion(outputs, data.y)
                loss.backward()

                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip)

                self.optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            self.losses = np.append(self.losses, avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}, Loss: {avg_loss}')

        # Calculate final MSE, MAE, and MAPE
        predictions = []
        targets = []
        for data in train_loader:
            data = data.to('cuda')
            outputs = self.model(data)

            if self.model.use_graph_conv and not self.model.use_temp_conv:
                outputs = outputs.permute(0, 2, 1)

            outputs = torch.split(outputs.cpu().detach(), 1, 0)
            outputs = np.array([o.squeeze(0).numpy() for o in outputs])
            outputs = np.concatenate(outputs)
            predictions.extend(outputs)

            ground_truth = torch.split(data.y.cpu().detach(), 1, 0)
            ground_truth = np.array([t.squeeze(0).numpy()
                                    for t in ground_truth])
            ground_truth = np.concatenate(ground_truth)
            targets.extend(ground_truth)

        targets = torch.Tensor(targets)
        predictions = torch.Tensor(predictions)

        mse = calc_err(targets, predictions, 'mse')
        mae = calc_err(targets, predictions, 'mae')
        mape_val = calc_err(targets, predictions, 'mape')

        print(
            f"Final Training Metrics, by Location (Houston, North, Panhandle):" +
            f"\nMSE: {mse}\nMAE: {mae}\nMAPE: {mape_val}%")

    def test(self, test_loader):
        self.model.eval()
        test_loss = 0.0
        predictions = []
        targets = []

        with torch.no_grad():
            for data in test_loader:
                data = data.to('cuda')
                outputs = self.model(data)
                if self.model.use_graph_conv and not self.model.use_temp_conv:
                    outputs = outputs.permute(0, 2, 1)

                loss = self.criterion(outputs, data.y)

                test_loss += loss.item()

                outputs = torch.split(outputs.cpu().detach(), 1, 0)
                outputs = np.array([o.squeeze(0).numpy() for o in outputs])
                outputs = np.concatenate(outputs)
                predictions.extend(outputs)

                ground_truth = torch.split(data.y.cpu().detach(), 1, 0)
                ground_truth = np.array([t.squeeze(0).numpy()
                                        for t in ground_truth])
                ground_truth = np.concatenate(ground_truth)
                targets.extend(ground_truth)

        test_loss /= len(test_loader)
        print(f'Test Loss: {test_loss}')

        predictions = torch.Tensor(predictions)
        targets = torch.Tensor(targets)

        mse = calc_err(targets, predictions, 'mse')
        mae = calc_err(targets, predictions, 'mae')
        mape_val = calc_err(targets, predictions, 'mape')

        print(
            f"Final Testing Metrics, by Location (Houston, North, Panhandle):" +
            f"\nMSE: {mse}\nMAE: {mae}\nMAPE: {mape_val}%")

        return mse, mae, mape_val
