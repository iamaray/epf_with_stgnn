import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import yaml
from sklearn.preprocessing import StandardScaler

from data_processing.transformations_funcs import MADStandardizer, ArcsinhTransformer
from data_processing.processing_classes import *
from data_processing.processor import PreprocessData
from data_processing.dataset_constructors import DatasetConstructor
from .trainers.curriculum_trainer import PredLenCurriculumTrainer
from .stgnn.model import STGNN, construct_STGNN
from .stgnn.stgnn_ats import STGNN_ATS, construct_ats_stgnn

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
