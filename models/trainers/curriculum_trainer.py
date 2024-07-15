import torch
import torch.optim as optim
import numpy as np

from trainers.error_metrics import *


class PredLenCurriculumTrainer:
    def __init__(
            self,
            models,
            epochs=60,
            lr=1e-3,
            weight_decay=1e-4,
            grad_clip=5,
            noise_mult=[0.05, 0.10, 0.15, 0.20, 0.0],
            criterion=masked_mae):

        self.models = models

        self.optimizers = [optim.Adam(
            m.parameters(), lr=lr, weight_decay=weight_decay) for m in models]
        # self.optimizer = optim.Adam(
        #     self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.epochs = epochs
        self.criterion = criterion
        self.losses = np.array([])
        self.grad_clip = grad_clip
        self.trained_model = None
        self.noise_mult = noise_mult

    def train(self, curriculum_loader):
        for i, (train_loader, test_loader) in enumerate(curriculum_loader):
            model = self.models[i]
            optimizer = self.optimizers[i]
            model.train()

            prev_state = None
            curr_state = model.state_dict()
            # load the weights from previous model that match shape
            # For pred length curriculum, this will be all the weights
            # except that of output_conv_2.
            if i > 0:
                prev_state = self.models[i - 1].state_dict()

                # filtered_state = {}
                # for k, v in prev_state.items():
                #   if curr_state[k].shape == prev_state[k].shape:
                #     filtered_state[k] = v
                #   else:

                filtered_state = {k: v for k, v in prev_state.items(
                ) if curr_state[k].shape == prev_state[k].shape}
                print("here:", len(prev_state.keys()) -
                      len(filtered_state.keys()))
                curr_state.update(filtered_state)
                model.load_state_dict(curr_state)

            print(f"\n======================TRAINING ON DATASET {
                  i}======================")
            for epoch in range(self.epochs):
                running_loss = 0.0
                for data in train_loader:
                    noise = torch.Tensor(
                        np.random.multivariate_normal(
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
                    # loss = self.criterion(outputs, data.y)
                    loss.backward()

                    if self.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), self.grad_clip)

                    optimizer.step()
                    running_loss += loss.item()

                avg_loss = running_loss / len(train_loader)
                self.losses = np.append(self.losses, avg_loss)

                # print(f'Epoch {epoch+1}, Loss: {avg_loss}')

                if (epoch + 1) % 10 == 0:
                    print(f'Epoch {epoch+1} of Dataset {i}, Loss: {avg_loss}')

            self.trained_model = self.models[i]
            self.test(test_loader)

        # Calculate final MSE, MAE, and MAPE
        predictions = []
        targets = []
        for data in train_loader:
            data = data.to('cuda')
            outputs = self.trained_model(data)

            if self.trained_model.use_graph_conv and not self.trained_model.use_temp_conv:
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
