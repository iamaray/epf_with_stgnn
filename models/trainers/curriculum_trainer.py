import torch
import torch.optim as optim
import numpy as np
import scipy as sc
import os
import json

from .evaluation import masked_mae, masked_avg_loc_diff, calc_err, GraphContinuityLoss


class PredLenCurriculumTrainer:
    def __init__(
            self,
            models,
            epochs=60,
            lr=1e-3,
            weight_decay=1e-4,
            grad_clip=5,
            noise_mult=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            pred_criterion=masked_mae,
            beta_ats=0.25,
            te_history=10,
            criterion_quantile=0.0):

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.models = [model.to(self.device) for model in models]
        self.criterion_quantile = criterion_quantile
        self.beta_ats = beta_ats
        self.optimizers = [optim.Adam(
            m.parameters(), lr=lr, weight_decay=weight_decay) for m in models]
        self.epochs = epochs
        self.pred_criterion = pred_criterion
        self.ats_criterion = GraphContinuityLoss(beta_ats)
        self.forward_ats_criterion = GraphContinuityLoss(beta=beta_ats)
        self.diff_loss = masked_avg_loc_diff

        self.grad_clip = grad_clip
        self.trained_model = None
        self.noise_mult = noise_mult

        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)

    def save_tensors(self, tensors, filename):
        """Helper function to save tensors of any shape"""
        if isinstance(tensors, torch.Tensor):
            tensors = tensors.detach().cpu().numpy()
        elif isinstance(tensors, list):
            tensors = np.array([t.detach().cpu().numpy() if isinstance(
                t, torch.Tensor) else t for t in tensors])
        np.save(filename, tensors)

    def train(self, curriculum_loader, use_ats):
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
                f"\n======================TRAINING ON DATASET {i}======================")
            for epoch in range(self.epochs):
                running_pred_loss = torch.tensor([0.0]).to(self.device)
                running_ats_loss = torch.tensor([0.0]).to(self.device)
                running_forward_loss = torch.tensor([0.0]).to(self.device)

                for data in train_loader:
                    data = data.to(self.device)

                    if self.noise_mult[i] > 0:
                        # Handle noise addition for any input shape
                        noise_shape = list(
                            data.x.shape[:-1]) + [data.x.shape[-1]]
                        noise = torch.tensor(
                            np.random.multivariate_normal(
                                mean=np.zeros((data.x.shape[-1])),
                                cov=np.eye(data.x.shape[-1]),
                                size=noise_shape[:-1])).to(self.device)

                        data.x = data.x + (self.noise_mult[i] * noise)

                    optimizer.zero_grad()

                    ats_loss = torch.tensor([0.0]).to(self.device)
                    forward_loss = torch.tensor([0.0]).to(self.device)

                    outputs = None
                    ats = None
                    forward_ats = None
                    if use_ats:
                        if model.f_ats:
                            outputs, ats, forward_ats = model(data)
                            ats_loss = self.ats_criterion(ats)
                        else:
                            outputs, ats, _ = model(data)
                            ats_loss = self.ats_criterion(ats)
                    else:
                        outputs, _, _ = model(data)

                    
                    pred_loss = self.pred_criterion(
                        outputs, data.y, quantile=self.criterion_quantile)

                    loss_locational = self.diff_loss(
                        outputs, data.y, beta=self.beta_ats)

                    loss = pred_loss + ats_loss + forward_loss + loss_locational
                    loss.backward()

                    if self.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), self.grad_clip)

                    optimizer.step()

                    running_pred_loss += pred_loss.item()
                    running_ats_loss += ats_loss.item()
                    running_forward_loss += forward_loss.item()

                avg_pred_loss = running_pred_loss / len(train_loader)
                avg_ats_loss = running_ats_loss / len(train_loader)
                avg_forward_loss = running_forward_loss / len(train_loader)
                print(
                    f'Epoch {epoch+1} of Dataset {i}, Loss: {avg_pred_loss.item(), avg_ats_loss.item(), avg_forward_loss.item()}')
                if (epoch + 1) % 10 == 0:
                    print(
                        f'Epoch {epoch+1} of Dataset {i}, Loss: {avg_pred_loss.item(), avg_ats_loss.item(), avg_forward_loss.item()}')

            self.trained_model = self.models[i]
            test_metrics = self.test(test_loader, use_ats=use_ats)

            # Save model state dict
            torch.save(self.trained_model.state_dict(),
                       f'results/model_dataset_{i}.pt')

        # Calculate and save final predictions
        predictions = []
        targets = []
        for data in train_loader:
            data = data.to(self.device)
            outputs, _, _ = self.trained_model(data)

            # Keep original tensor dimensions until final concatenation
            pred_batch = outputs.detach().cpu()
            target_batch = data.y.detach().cpu()

            predictions.append(pred_batch)
            targets.append(target_batch)

        # Concatenate along batch dimension (dim=0)
        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)

        # Save predictions and targets preserving original shapes
        self.save_tensors(predictions, 'results/final_predictions.npy')
        self.save_tensors(targets, 'results/final_targets.npy')

        mae = calc_err(targets, predictions, 'mae')

        metrics = {
            'final_mae': mae.tolist() if isinstance(mae, torch.Tensor) else mae,
            'prediction_shape': list(predictions.shape),
            'target_shape': list(targets.shape)
        }

        with open('results/final_metrics.json', 'w') as f:
            json.dump(metrics, f)

        # Save final model weights after all training is complete
        torch.save(self.trained_model.state_dict(), 'results/final_model.pt')

        print(
            f"Final Training Metrics, by Location (Houston, North, Panhandle):" +
            f"\nMSE: NA\nMAE: {mae}\nMAPE: NA")

    def test(self, test_loader, use_ats):
        self.trained_model.eval()
        test_loss = 0.0
        predictions = []
        targets = []

        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                outputs, _, _ = self.trained_model(data)

                loss = self.pred_criterion(outputs, data.y)
                test_loss += loss.item()

                # Keep original tensor dimensions
                pred_batch = outputs.detach().cpu()
                target_batch = data.y.detach().cpu()

                predictions.append(pred_batch)
                targets.append(target_batch)

        # Concatenate along batch dimension
        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)

        test_loss /= len(test_loader)
        print(f'Test Loss: {test_loss}')

        # Save test predictions and targets preserving shapes
        self.save_tensors(predictions, 'results/test_predictions.npy')
        self.save_tensors(targets, 'results/test_targets.npy')

        mae = calc_err(targets, predictions, 'mae')

        metrics = {
            'test_loss': test_loss,
            'test_mae': mae.tolist() if isinstance(mae, torch.Tensor) else mae,
            'prediction_shape': list(predictions.shape),
            'target_shape': list(targets.shape)
        }

        with open('results/test_metrics.json', 'w') as f:
            json.dump(metrics, f)

        print(
            f"Final Testing Metrics, by Node:" +
            f"\nMSE: NA\nMAE: {mae}\nMAPE: NA")

        return 0.0, mae, 0.0
