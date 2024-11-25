import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import yaml
from sklearn.preprocessing import StandardScaler

from .transformations_funcs import MADStandardizer, ArcsinhTransformer
from .processing_classes import *
from .processor import PreprocessData
from .dataset_constructors import *
from .dataset_constructors import DatasetConstructor


transformation_options = {
    'mad_std': MADStandardizer(), 'arcsinh_norm': ArcsinhTransformer(), 'std': StandardScaler()}


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

        self.preproc_pipeline = Pipeline(self.transforms)

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
        self.learned_ats_params = self.config.get(
            'model_params', {}).get('forward_ats_params', {})
        if not isinstance(self.learned_ats_params, dict):
            raise ValueError(
                "Expected 'learned_ats_params' to be a dictionary.")

        self.model = None
        self.trainer = None

    def preprocess(self, data_path, generate_hour_dummies=True, targets=[], aux_feats=[]):
        """ Produce a preprocessed dataset """
        df = None
        try:
            df = pd.read_csv(data_path)
        except:
            raise NotImplementedError("For now, please provide a csv file.")

        feats = np.array([])

        transformed_targets = self.preproc_pipeline.fit_transform(df[targets])
        hours, _ = transformed_targets.shape

        hour_dummies = np.array([])
        if generate_hour_dummies:
            hour_dummies = np.arange(hours) & 24
            hour_dummies = hour_dummies.reshape(-1, 1)

        feats = []
        for i, feat_list in enumerate(aux_feats):
            aux = self.preproc_pipeline.fit_transform(df[feat_list])
            target = transformed_targets[:, i].reshape(-1, 1)

            components = [target, aux]
            if generate_hour_dummies:
                components.append(hour_dummies)

            curr_feats = np.concatenate(
                components, axis=-1)
            feats.append(curr_feats)


        feats = np.array(feats)
        feats = torch.permute(torch.Tensor(feats), (2, 0, 1))

    def train(self):
        pass

    def predict(self):
        pass

    def evaluate(self):
        pass
