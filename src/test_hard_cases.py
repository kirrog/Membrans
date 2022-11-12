from tensorflow import keras

from src.utils.config_loader import model_path
from src.utils.config_loader import model_path_logs
from src.utils.config_loader import output_data_dir
from src.utils.config_loader import one_test_data_dir

from src.clearer.datasets_loader import load_clearer_dataset_tests, load_test_dataset_from_images_from
from src.clearer.models.clearer_model_attention_u_net import clearer_model_attention_u_net
from src.clearer.models.clearer_model_r2u_net import clearer_model_r2u_net
from src.clearer.models.clearer_model_ru_net import clearer_model_ru_net
from src.clearer.models.clearer_model_u_net import clearer_model_u_net
from src.utils.matrix2png_saver import transform_results

logs = True
if logs:
    model = clearer_model_r2u_net()
    model.load_weights(model_path_logs)
else:
    model = keras.models.load_model(model_path)

predicts = load_test_dataset_from_images_from(one_test_data_dir)
res = model.predict(x=predicts, batch_size=2)
transform_results(res, output_data_dir)
