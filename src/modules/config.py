import json

config_path = 'inference_config.json'

with open(config_path) as f:
    dictionary = json.load(f)
batch_size = dictionary["batch_size"]
clearer_model_path = dictionary["clearer_model_path"]
detector_model_path = dictionary["detector_model_path"]
converter_model_path = dictionary["converter_model_path"]
