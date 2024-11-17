import torch
import torch.optim as optim
import numpy as np
from .evaluation import *


class PredLenCurriculumTrainer:
    """
    Trainer class for training and evaluating multiple PyTorch models with prediction length curriculum.

    Parameters:
        models (list of torch.nn.Module): List of models to be trained.
        epochs (int): Number of training epochs. Default is 60.
        lr (float): Learning rate for the optimizer. Default is 1e-3.
        weight_decay (float): Weight decay (L2 penalty) for the optimizer. Default is 1e-4.
        grad_clip (float): Gradient clipping value. Default is 5.
        noise_mult (list of float): Multipliers for adding noise to the data. Default is [0.05, 0.10, 0.15, 0.20, 0.0].
        criterion (function): Loss function to use. Default is masked_mae.

    Attributes:
        models (list of torch.nn.Module): List of models to be trained.
        optimizers (list of torch.optim.Optimizer): List of optimizers for each model.
        epochs (int): Number of training epochs.
        criterion (function): Loss function to use.
        losses (numpy.ndarray): Array to store training losses.
        grad_clip (float): Gradient clipping value.
        trained_model (torch.nn.Module): The most recently trained model.
        noise_mult (list of float): Multipliers for adding noise to the data.
    """

    def __init__(self, models, epochs=60, lr=1e-3, weight_decay=1e-4, grad_clip=5, noise_mult=[0.05, 0.10, 0.15, 0.20, 0.0], criterion=masked_mae):
        self.models = models
        self.optimizers = [optim.Adam(
            m.parameters(), lr=lr, weight_decay=weight_decay) for m in models]
        self.epochs = epochs
        self.criterion = criterion
        self.losses = np.array([])
        self.grad_clip = grad_clip
        self.trained_model = None
        self.noise_mult = noise_mult

    def train(self, curriculum_loader):
        """
        Trains the models using the provided curriculum data loaders.

        Parameters:
            curriculum_loader (list of tuples): List of tuples containing training and testing data loaders.

        Returns:
            None
        """
        for i, (train_loader, test_loader) in enumerate(curriculum_loader):
            model = self.models[i]
            optimizer = self.optimizers[i]
            model.train()

            prev_state = None
            curr_state = model.state_dict()
            if i > 0:
                prev_state = self.models[i - 1].state_dict()
                filtered_state = {k: v for k, v in prev_state.items(
                ) if curr_state[k].shape == prev_state[k].shape}
                curr_state.update(filtered_state)
                model.load_state_dict(curr_state)

            print(
                f"\n====================== TRAINING ON DATASET {i} ======================")
            for epoch in range(self.epochs):
                running_loss = 0.0
                for data in train_loader:
                    noise = torch.Tensor(np.random.multivariate_normal(
                        mean=np.zeros((data.x.shape[-1])),
                        cov=np.eye(data.x.shape[-1]),
                        size=data.x.shape[:-1])).to('cuda')

                    data = data.to('cuda')
                    data.x = data.x + (self.noise_mult[i] * noise)
                    optimizer.zero_grad()
                    outputs = model(data)
                    if model.use_graph_conv and not model.use_temp_conv:
                        outputs = outputs.permute(0, 2, 1)

                    loss = self.criterion(outputs, data.y, 0.0)
                    loss.backward()

                    if self.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), self.grad_clip)

                    optimizer.step()
                    running_loss += loss.item()

                avg_loss = running_loss / len(train_loader)
                self.losses = np.append(self.losses, avg_loss)

                if (epoch + 1) % 10 == 0:
                    print(
                        f'Epoch {epoch + 1} of Dataset {i}, Loss: {avg_loss}')

            self.trained_model = self.models[i]
            self.test(test_loader)

        self._final_metrics(train_loader)

    def test(self, test_loader):
        """
        Evaluates the most recently trained model using the provided test data loader.

        Parameters:
            test_loader (torch.utils.data.DataLoader): DataLoader for the test data.

        Returns:
            tuple: MSE, MAE, and MAPE values for the test set.
        """
        self.trained_model.eval()
        test_loss = 0.0
        predictions = []
        targets = []

        with torch.no_grad():
            for data in test_loader:
                data = data.to('cuda')
                outputs = self.trained_model(data)
                if self.trained_model.use_graph_conv and not self.trained_model.use_temp_conv:
                    outputs = outputs.permute(0, 2, 1)

                loss = self.criterion(outputs, data.y)
                test_loss += loss.item()

                predictions.extend(self._process_outputs(outputs))
                targets.extend(self._process_outputs(data.y))

        test_loss /= len(test_loader)
        print(f'Test Loss: {test_loss}')

        mse, mae, mape_val = self._calculate_errors(targets, predictions)
        print(
            f"Final Testing Metrics, by Location (Houston, North, Panhandle):\nMSE: {mse}\nMAE: {mae}\nMAPE: {mape_val}%")

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
            outputs = self.trained_model(data)
            if self.trained_model.use_graph_conv and not self.trained_model.use_temp_conv:
                outputs = outputs.permute(0, 2, 1)
            predictions.extend(self._process_outputs(outputs))
            targets.extend(self._process_outputs(data.y))

        mse, mae, mape_val = self._calculate_errors(targets, predictions)
        print(
            f"Final Training Metrics, by Location (Houston, North, Panhandle):\nMSE: {mse}\nMAE: {mae}\nMAPE: {mape_val}%")

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
