import torch
import torch.optim as optim
import numpy as np
import scipy as sc

from .evaluation import masked_mae, masked_avg_loc_diff, calc_err


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

        self.models = models
        self.criterion_quantile = criterion_quantile
        self.beta_ats = beta_ats
        self.optimizers = [optim.Adam(
            m.parameters(), lr=lr, weight_decay=weight_decay) for m in models]
        # self.optimizer = optim.Adam(
        #     self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.epochs = epochs
        self.pred_criterion = pred_criterion
        self.ats_criterion = GraphContinuityLoss(beta_ats)
        # TODO: implement TE loss
        self.forward_ats_criterion = GraphContinuityLoss(beta=beta_ats)
        self.diff_loss = masked_avg_loc_diff

        # self.losses = np.array([])
        self.grad_clip = grad_clip
        self.trained_model = None
        self.noise_mult = noise_mult

    def train(self, curriculum_loader, use_ats):
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
                # print("here:", len(prev_state.keys()) - len(filtered_state.keys()))
                curr_state.update(filtered_state)
                model.load_state_dict(curr_state)

            print(
                f"\n======================TRAINING ON DATASET {i}======================")
            for epoch in range(self.epochs):
                running_pred_loss = torch.Tensor([0.0]).to('cuda')
                running_ats_loss = torch.Tensor([0.0]).to('cuda')
                running_forward_loss = torch.Tensor([0.0]).to('cuda')

                for data in train_loader:
                    noise = torch.Tensor(
                        np.random.multivariate_normal(
                            mean=np.zeros((data.x.shape[-1])),
                            cov=np.eye(data.x.shape[-1]),
                            size=data.x.shape[:-1])).to('cuda')

                    data = data.to('cuda')
                    data.x = data.x + (self.noise_mult[i] * noise)
                    optimizer.zero_grad()

                    ats_loss = torch.Tensor([0.0]).to('cuda')
                    forward_loss = torch.Tensor([0.0]).to('cuda')

                    outputs = None
                    ats = None
                    forward_ats = None
                    if use_ats:
                        # print(data.x.shape)
                        if model.f_ats:
                            outputs, ats, forward_ats = model(data)
                            ats_loss = self.ats_criterion(ats)
                            # forward_loss = self.ats_criterion(forward_ats) - self.forward_ats_criterion(torch.abs(forward_ats))
                        else:
                            outputs, ats, _ = model(data)
                            ats_loss = self.ats_criterion(ats)
                            # print(ats_loss)
                        # print('here:', forward_ats.cpu().detach().numpy() < 0)
                    else:
                        outputs, _, _ = model(data)

                    # if model.use_graph_conv and not model.use_temp_conv:
                    #     outputs = outputs.permute(0, 2, 1)

                    # print(outputs.shape, data.y.shape)
                    pred_loss = self.pred_criterion(
                        outputs, data.y, quantile=self.criterion_quantile)

                    loss_locational = self.diff_loss(
                        outputs, data.y, beta=self.beta_ats)

                    loss = pred_loss + ats_loss + forward_loss + loss_locational

                    # loss = self.criterion(outputs, data.y)
                    # print('here')
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

                # print(f'Epoch {epoch+1}, Loss: {avg_loss}')

                if (epoch + 1) % 10 == 0:
                    print(
                        f'Epoch {epoch+1} of Dataset {i}, Loss: {avg_pred_loss.item(), avg_ats_loss.item(), avg_forward_loss.item()}')

            self.trained_model = self.models[i]
            self.test(test_loader, use_ats=use_ats)

        # Calculate final MSE, MAE, and MAPE
        predictions = []
        targets = []
        for data in train_loader:
            data = data.to('cuda')
            data = data
            outputs, _, _ = self.trained_model(data)

            # if self.trained_model.use_graph_conv and not self.trained_model.use_temp_conv:
            #     outputs = outputs.permute(0, 2, 1)

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

        # mse = calc_err(targets, predictions, 'mse')
        mae = calc_err(targets, predictions, 'mae')
        # mape_val = calc_err(targets, predictions, 'mape')

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
                data = data.to('cuda')
                outputs, _, _ = self.trained_model(data)

                # if self.trained_model.use_graph_conv and not self.trained_model.use_temp_conv:
                #     outputs = outputs.permute(0, 2, 1)

                loss = self.pred_criterion(outputs, data.y)

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

        # mse = calc_err(targets, predictions, 'mse')
        mae = calc_err(targets, predictions, 'mae')
        # mape_val = calc_err(targets, predictions, 'mape')

        print(
            f"Final Testing Metrics, by Location (Houston, North, Panhandle):" +
            f"\nMSE: NA\nMAE: {mae}\nMAPE: NA")

        return 0.0, mae, 0.0
