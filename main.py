# from sklearn.pipeline import Pipeline

# from data_processing.processing_classes import *
# from data_processing.transformations_funcs import MADStandardScaler, ArcsinhTransformer
# from data_processing.dataset_constructors import DatasetConstructor
# from data_processing.processor import PreprocessData
from models.pipelines import STGNNPipeline


def main():
    data_path = 'data/combined_dataset.csv'

    pipeline = STGNNPipeline('Configs/stgnn_config.yml')

    pipeline.preprocess(data_path=data_path, targets=[
                        f'LMP_DIFF_{i}' for i in range(29)], aux_feats=[[f'S_{i}'] for i in range(29)])


if __name__ == "__main__":
    main()
