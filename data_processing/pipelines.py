import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import yaml

from .transformations_funcs import MADStandardScaler, ArcsinhTransformer
from .processing_classes import *
from .processor import PreprocessData
from .dataset_constructors import *
from .dataset_constructors import DatasetConstructor

transformation_options = {
    'mad': MADStandardScaler(), 'arcinsh': ArcsinhTransformer()}


class STGNNPipeline:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

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

    def preprocess(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass

    def evaluate(self):
        pass
