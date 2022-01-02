# from src.generator.model.generator_model_u_net import generator_model_u_net, save_generator_model
# from src.generator.trainer.trainer import train_generator_model
#
# model = generator_model_u_net()
# train_generator_model(model)
# save_generator_model(model)
from src.generator.dataset_generator_providers.dataset_generator_providers import generator_dataset_pair_creater
from src.utils.config_loader import test_data

test_dataset = generator_dataset_pair_creater(test_data)
