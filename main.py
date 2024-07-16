from models.stgnn.model import construct_STGNN
from models.trainers.curriculum_trainer import PredLenCurriculumTrainer

def run_stgnn_on_curriculum(curriculum_data, epochs=60):
    models = []
    
    for tup in curriculum_data:
        tr, _ = tup
        models.append(construct_STGNN(tr.dataset[0], dropout_factor=0.3))
        
    curr_trainer = PredLenCurriculumTrainer(models=models, epochs=epochs)
    
    curr_trainer.train(curriculum_data)
    
    return curr_trainerl.trained_model
    