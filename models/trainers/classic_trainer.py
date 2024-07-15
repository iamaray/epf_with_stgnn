import torch
import numpy as np
import torch.optim as optim
from trainers.error_metrics import *


class Trainer:
    """
    Trainer class for training and evaluating a PyTorch model.

    Parameters:
        model (torch.nn.Module): The model to be trained.
        epochs (int): Number of training epochs. Default is 60.
        lr (float): Learning rate for the optimizer. Default is 1e-3.
        weight_decay (float): Weight decay (L2 penalty) for the optimizer. Default is 1e-4.
        grad_clip (float): Gradient clipping value. Default is 5.
        criterion (function): Loss function to use. Default is masked_mae.

    Attributes:
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        epochs (int): Number of training epochs.
        criterion (function): Loss function to use.
        losses (numpy.ndarray): Array to store training losses.
        grad_clip (float): Gradient clipping value.
    """

    def __init__(self, model, epochs=60, lr=1e-3, weight_decay=1e-4, grad_clip=5, criterion=masked_mae):
        self.model = model
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.epochs = epochs
        self.criterion = criterion
        self.losses = np.array([])
        self.grad_clip = grad_clip

    def train(self, train_loader):
        """
        Trains the model using the provided training data loader.

        Parameters:
            train_loader (torch.utils.data.DataLoader): DataLoader for the training data.

        Returns:
            None
        """
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

        self._final_metrics(train_loader)

    def test(self, test_loader):
        """
        Evaluates the model using the provided test data loader.

        Parameters:
            test_loader (torch.utils.data.DataLoader): DataLoader for the test data.

        Returns:
            tuple: MSE, MAE, and MAPE values for the test set.
        """
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

                predictions.extend(self._process_outputs(outputs))
                targets.extend(self._process_outputs(data.y))

        test_loss /= len(test_loader)
        print(f'Test Loss: {test_loss}')

        mse, mae, mape_val = self._calculate_errors(targets, predictions)
        print(f"Final Testing Metrics, by Location (Houston, North, Panhandle):\nMSE: {
              mse}\nMAE: {mae}\nMAPE: {mape_val}%")

        return mse, mae, mape_val

    def _final_metrics(self, loader):
        """
        Calculates final MSE, MAE, and MAPE metrics after training.

        Parameters:
            loader (torch.utils.data.DataLoader): DataLoader for the data.

        Returns:
            None
        """
        predictions = []
        targets = []
        for data in loader:
            data = data.to('cuda')
            outputs = self.model(data)
            if self.model.use_graph_conv and not self.model.use_temp_conv:
                outputs = outputs.permute(0, 2, 1)
            predictions.extend(self._process_outputs(outputs))
            targets.extend(self._process_outputs(data.y))

        mse, mae, mape_val = self._calculate_errors(targets, predictions)
        print(f"Final Training Metrics, by Location (Houston, North, Panhandle):\nMSE: {
              mse}\nMAE: {mae}\nMAPE: {mape_val}%")

    def _process_outputs(self, outputs):
        """
        Processes model outputs for metric calculation.

        Parameters:
            outputs (torch.Tensor): Model outputs.

        Returns:
            list: Processed outputs.
        """
        outputs = torch.split(outputs.cpu().detach(), 1, 0)
        outputs = np.array([o.squeeze(0).numpy() for o in outputs])
        outputs = np.concatenate(outputs)
        return outputs

    def _calculate_errors(self, targets, predictions):
        """
        Calculates MSE, MAE, and MAPE metrics.

        Parameters:
            targets (list): Ground truth targets.
            predictions (list): Model predictions.

        Returns:
            tuple: MSE, MAE, and MAPE values.
        """
        targets = torch.Tensor(targets)
        predictions = torch.Tensor(predictions)
        mse = calc_err(targets, predictions, 'mse')
        mae = calc_err(targets, predictions, 'mae')
        mape_val = calc_err(targets, predictions, 'mape')
        return mse, mae, mape_val
