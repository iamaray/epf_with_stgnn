import torch
import numpy as np
import pandas as pd
from models.fourier_gnn.model import FGN, construct_fgn
from models.pipelines import FourierGNNPipeline
from models.trainers.curriculum_trainer import PredLenCurriculumTrainer
import itertools


def main():
    pipeline = FourierGNNPipeline('data/combined_dataset.csv')
    multi_step_data = pipeline(
        target_feats=[f'LMP_{i}' for i in range(30)], pred_hours_list=[6, 12, 18, 24])

    models = []
    for tr, te in multi_step_data:
        data = next(iter(tr))
        models.append(construct_fgn(data))

    trainer = PredLenCurriculumTrainer(models=models, epochs=60)
    trainer.train(curriculum_loader=multi_step_data, use_ats=False)

    trained_model = pipeline.load_model(model_path='results/final_model.pt')

    pipeline.evaluate(model=trained_model)


if __name__ == "__main__":
    main()
