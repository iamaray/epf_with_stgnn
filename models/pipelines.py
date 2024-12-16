from .fourier_gnn.model import FGN
from .trainers.evaluation import conditional_mae_gte_quantile
from torch_geometric.loader import DataLoader
from .stgnn.stgnn_ats import STGNN_ATS, construct_ats_stgnn
from .stgnn.model import STGNN, construct_STGNN
from .trainers.curriculum_trainer import PredLenCurriculumTrainer
from data_processing.dataset_constructors import DatasetConstructor
from data_processing.processor import PreprocessData
from data_processing.processing_classes import *
from data_processing.transformations_funcs import MADStandardizer, ArcsinhTransformer
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
import yaml
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

transformation_options = {
    'mad_std': MADStandardizer(), 'arcsinh_norm': ArcsinhTransformer(), 'std': StandardScaler()}


class Preprocessor:
    """ Takes in a Pandas DataFrame and outputs a curriculum train/test set. """

    def __init__(self):
        pass


class TrainWrapper:
    """ Takes in a train/test set, initializes the model(s), and performs training/evaluation. """

    def __init__(self):
        pass


class Visualizer:
    """ Takes your model(s) and produces various plots. """

    def __init__(self):
        pass


class STGNNPipeline:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.time_dummies = self.config['preprocessing']['time_dummies']

        self.transforms = []
        transform_funcs = self.config.get(
            'preprocessing', {}).get('transform_funcs', [])
        for transform in transform_funcs:
            name = transform.get('name')
            if name in transformation_options:
                self.transforms.append((name, transformation_options[name]))
            else:
                raise ValueError(
                    f"Transformation function '{name}' is not supported.")

        self.preproc_pipeline_targets = Pipeline(self.transforms)
        self.preproc_pipeline_aux = Pipeline(self.transforms)

        self.curriculum_type = self.config['dataset_params']['curriculum_type']
        self.curriculum = self.config['dataset_params']['curriculum']
        self.batch_size = self.config['dataset_params']['batch_size']
        self.window_hours = self.config['dataset_params']['window_hours']
        self.step_hours = self.config['dataset_params']['step_hours']
        self.split_hour = self.config['dataset_params']['split_hour']
        self.copula_adj = self.config['dataset_params']['copula_adj']

        self.model_params = self.config.get(
            'model_params', {}).get('ots_model_params', {})
        print(type(self.model_params))
        if not isinstance(self.model_params, dict):
            raise ValueError("Expected 'ots_model_params' to be a dictionary.")

        self.use_learned_ats = self.config['model_params']['use_learned_ats']
        self.learned_ats_params = self.config.get(
            'model_params', {}).get('learned_ats_params', {})
        if not isinstance(self.learned_ats_params, dict):
            raise ValueError(
                "Expected 'learned_ats_params' to be a dictionary.")

        self.use_forward_ats = self.config['model_params']['use_forward_ats']
        self.forward_ats_params = self.config.get(
            'model_params', {}).get('forward_ats_params', {})
        if not isinstance(self.learned_ats_params, dict):
            raise ValueError(
                "Expected 'learned_ats_params' to be a dictionary.")

        self.trainer_params = self.config.get('training_params', {})

        self.dataset_constr = DatasetConstructor(
            curriculum_type=self.curriculum_type,
            curriculum=self.curriculum,
            batch_size=self.batch_size,
            copula_adj=self.copula_adj,
            window_hours=self.window_hours,
            step_hours=self.step_hours,
            split_hour=self.split_hour)

        self.model = None
        self.trainer = None

    def preprocess(self, data_path, generate_hour_dummies=True, targets=[], aux_feats=[]):
        """ Produce a preprocessed dataset """
        df = None
        try:
            df = pd.read_csv(data_path)
        except:
            raise NotImplementedError("For now, please provide a csv file.")

        # FEATURE EXTRACTION
        feats = np.array([])

        transformed_targets = self.preproc_pipeline_targets.fit_transform(
            df[targets])
        hours, _ = transformed_targets.shape

        hour_dummies = np.array([])
        if generate_hour_dummies:
            hour_dummies = np.arange(hours) & 24
            hour_dummies = hour_dummies.reshape(-1, 1)

        feats = []
        for i, feat_list in enumerate(aux_feats):
            aux = self.preproc_pipeline_aux.fit_transform(df[feat_list])
            target = transformed_targets[:, i].reshape(-1, 1)

            components = [target, aux]
            if generate_hour_dummies:
                components.append(hour_dummies)

            curr_feats = np.concatenate(
                components, axis=-1)
            feats.append(curr_feats)

        feats = np.array(feats)
        feats = torch.permute(torch.Tensor(feats), (2, 0, 1))
        print(feats.shape)
        # TRAIN/TEST PAIR CONSTRUCTION
        self.curr_data = self.dataset_constr(feats)

        # initialize the models on data
        self.models = []
        for tup in self.curr_data:
            tr, _ = tup

            self.models.append(construct_ats_stgnn(
                data=tr.dataset[0],
                ats_constr_params=self.learned_ats_params,
                fwd_ats_constr_params=self.forward_ats_params,
                ots_model_params=self.model_params,
                ats_model_params=self.model_params,
                forward_model_params=self.model_params,
                cats=self.use_learned_ats,
                f_ats=self.use_forward_ats
            ))

        # initialize the trainer on models
        self.trainer = PredLenCurriculumTrainer(
            models=self.models,
            epochs=self.trainer_params['epochs'],
            lr=self.trainer_params['lr'],
            weight_decay=self.trainer_params['grad_clip'],
            noise_mult=self.trainer_params['noise_mult'],
            pred_criterion=self.trainer_params['noise_mult'],
            beta_ats=self.trainer_params['beta_ats'],
            te_history=self.trainer_params['te_history'],
            criterion_quantile=self.trainer_params['criterion_quantile'])

    def train_test(self):
        self.trainer.train(self.curr_data, use_ats=self.use_learned_ats)

    def predict(self):
        pass

    def evaluate(self):
        # self.trainer.test()
        pass


class FourierGNNPipeline:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data = pd.read_csv(csv_path)
        self.scaler = StandardScaler()
        self.arcsinh = ArcsinhTransformer()
        self.proc_data = None
        self.pairs = []
        self.model = None
        self.raw_data = None
        self.in_sample_len = None

    def preprocess(self, target_feats, in_sample_percent=0.75):
        # Extract target features
        self.in_sample_len = int(len(self.data) * in_sample_percent)
        self.raw_data = self.data[target_feats].values
        data = self.raw_data[:self.in_sample_len]

        # Log transform
        data = self.scaler.fit_transform(data)
        # data = np.log1p(1 + np.abs(data))
        data = np.arcsinh(data)

        # Standardize

        # Arcsinh normalize
        # data = self.arcsinh.fit_transform(data)

        # Reshape to [num_variates, seq_len]
        # Assuming each row is a timestep and each column is a feature
        data = torch.tensor(data).T.float()
        self.proc_data = data

        # Check for NaN values
        print(f"Data contains NaN values: {torch.isnan(data).any().item()}")

        return data

    def form_training_pairs(self, window_hours, step_hours, pred_hours):
        """Form (x,y) pairs using sliding window over time steps"""
        if self.proc_data is None:
            raise ValueError(
                "Must preprocess data before forming training pairs")

        data = self.proc_data  # Data is already [num_variates, seq_len]
        num_steps = data.shape[1]
        pairs = []

        # Slide window over time steps
        for i in range(0, num_steps - window_hours - pred_hours + 1, step_hours):
            x = data[:, i:i+window_hours]
            y = data[:, i+window_hours:i+window_hours+pred_hours]
            pairs.append((x, y))

        self.pairs = pairs
        return pairs

    def to_dataloader(self, train_percent, batch_size=32):
        """Convert (x,y) pairs to train/test dataloaders using PyG Data objects"""
        if not self.pairs:
            raise ValueError(
                "Must form training pairs before creating dataloader")

        # Split into train/test
        n_train = int(len(self.pairs) * train_percent)
        train_pairs = self.pairs[:n_train]
        test_pairs = self.pairs[n_train:]

        # Create PyG Data objects
        train_data_list = []
        for x, y in train_pairs:
            # Reshape x and y to be [batch, num_nodes, seq_len]
            x = x.unsqueeze(0)  # Add batch dimension
            y = y.unsqueeze(0)  # Add batch dimension
            data = Data(x=x, y=y)
            train_data_list.append(data)

        test_data_list = []
        for x, y in test_pairs:
            # Reshape x and y to be [batch, num_nodes, seq_len]
            x = x.unsqueeze(0)  # Add batch dimension
            y = y.unsqueeze(0)  # Add batch dimension
            data = Data(x=x, y=y)
            test_data_list.append(data)

        # Create dataloaders with explicit batch size
        train_loader = DataLoader(
            train_data_list, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(
            test_data_list, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

    def load_model(self, model_path):
        """Load a trained FGN model from a .pt file"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        # Load state dict to CPU
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))

        # Initialize model and load weights
        self.model = FGN(pre_length=24, embed_size=32, feature_size=30, seq_length=504, hidden_size=32,
                         hidden_size_factor=1, sparsity_threshold=0.01, hard_thresholding_fraction=1)  # Initialize with default params
        self.model.load_state_dict(state_dict)
        self.model.eval()  # Set to evaluation mode

        return self.model

    def evaluate(self, model, criterion=conditional_mae_gte_quantile, window_hours=504):
        """Evaluate model on out-of-sample data"""
        if self.raw_data is None or self.in_sample_len is None:
            raise ValueError("Must run preprocess() first")

        # Get out of sample data
        out_sample_data = self.raw_data[self.in_sample_len:]

        # Apply same normalization as training
        norm_data = self.scaler.transform(out_sample_data)
        # norm_data = np.log1p(1 + np.abs(norm_data))
        norm_data = np.arcsinh(norm_data)
        norm_data = torch.tensor(norm_data).T.float()

        # Create evaluation pairs
        eval_pairs = []
        num_steps = norm_data.shape[1]
        pred_hours = 24  # Default prediction horizon

        for i in range(0, num_steps - window_hours - pred_hours + 1):
            x = norm_data[:, i:i+window_hours]
            y = norm_data[:, i+window_hours:i+window_hours+pred_hours]
            eval_pairs.append((x, y))

        # Evaluate model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        all_preds = []
        all_targets = []
        all_inputs = []

        with torch.no_grad():
            for x, y in eval_pairs:
                x = x.unsqueeze(0).to(device)  # Add batch dimension
                y = y.unsqueeze(0).to(device)

                # Get model predictions
                pred, _, _ = model(Data(x=x, y=y))

                # Move predictions to CPU and convert to numpy
                pred = pred.cpu().numpy()
                y = y.cpu().numpy()
                x = x.cpu().numpy()

                # Denormalize predictions and targets
                # Reshape to match scaler's expected input shape (n_samples, n_features)
                pred_denorm = np.sinh(pred.squeeze())  # Remove batch dimension
                pred_denorm = pred_denorm.T  # Transpose to (time, features)
                pred_denorm = self.scaler.inverse_transform(pred_denorm)
                pred_denorm = pred_denorm.T  # Transpose back to (features, time)

                y_denorm = np.sinh(y.squeeze())
                y_denorm = y_denorm.T
                y_denorm = self.scaler.inverse_transform(y_denorm)
                y_denorm = y_denorm.T

                x_denorm = np.sinh(x.squeeze())
                x_denorm = x_denorm.T
                x_denorm = self.scaler.inverse_transform(x_denorm)
                x_denorm = x_denorm.T

                all_preds.append(pred_denorm)
                all_targets.append(y_denorm)
                all_inputs.append(x_denorm)

        # Convert to arrays
        predictions = np.array(all_preds)
        targets = np.array(all_targets)
        inputs = np.array(all_inputs)

        # Calculate error metrics
        error = criterion(actuals=torch.tensor(targets),
                          model_output=torch.tensor(predictions), quantile=0.0)

        # Save results
        os.makedirs('evaluation_results', exist_ok=True)
        np.save('evaluation_results/predictions.npy', predictions)
        np.save('evaluation_results/targets.npy', targets)
        np.save('evaluation_results/inputs.npy', inputs)
        np.save('evaluation_results/error.npy', error.numpy())

        return error

    def __call__(self, target_feats, window_hours=504, step_hours=24, pred_hours_list=[24], train_percent=0.7, batch_size=32):
        """Run full pipeline and return curriculum dataset with varying prediction lengths"""
        # Preprocess data
        self.preprocess(target_feats)

        # Create curriculum dataset
        curriculum_data = []
        for pred_hours in pred_hours_list:
            # Form training pairs for this prediction length
            self.form_training_pairs(window_hours, step_hours, pred_hours)

            # Convert to dataloaders
            train_loader, test_loader = self.to_dataloader(
                train_percent, batch_size)

            # Add to curriculum
            curriculum_data.append((train_loader, test_loader))

        return curriculum_data
