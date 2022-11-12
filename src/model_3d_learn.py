from src.gen_3d.model.generator_model_detector import detector_model_getter
from src.gen_3d.trainer.detector_trainer import train_detector_model

model = detector_model_getter()
train_detector_model(model)

# train_dataset = generator_dataset_pair_creater(train_data)
# test_dataset = generator_dataset_pair_creater(test_data)
