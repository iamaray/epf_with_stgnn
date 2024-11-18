from sklearn.pipeline import Pipeline

from data_processing.processing_classes import *
from data_processing.transformations_funcs import MADStandardScaler, ArcsinhTransformer
from data_processing.dataset_constructors import DatasetConstructor
from data_processing.processor import PreprocessData

data_path = 'data/refilled.csv'

op_history = OperationHistory([])

mad_standardization = MADStandardScaler(logging=True, op_history=op_history)
arcsinh = ArcsinhTransformer(logging=True, op_history=op_history)
op_pipeline = Pipeline([mad_standardization, arcsinh])

target_signal_names = ['SP_Price_Houston', 'SP_Price_North',
                       'SP_Price_Panh', 'SP_Price_South', 'SP_Price_West']
general_aux_feat_names = ['ACTUAL_ERC_Solar',
                          'ACTUAL_ERC_Wind', 'ACTUAL_ERC_Load']

houston_aux_feat_names = [] + general_aux_feat_names
north_aux_feat_names = [] + general_aux_feat_names
panh_aux_feat_names = [] + general_aux_feat_names
south_aux_feat_names = [] + general_aux_feat_names
west_aux_feat_names = [] + general_aux_feat_names

aux_feats = [houston_aux_feat_names, north_aux_feat_names,
             panh_aux_feat_names, south_aux_feat_names, west_aux_feat_names]

feat_map = {}
for i, name in enumerate(target_signal_names):
    feat_map[name] = aux_feats[i]

processor = PreprocessData(op_list=op_pipeline)
processor(data_path, feat_map)
