import glob
import sys
import cv2
import numpy as np

from natsort import natsorted
from tensorflow import keras

patient_path = '../patient_data'
clearer_model_path = '../result_models/clearer_weights.h5'
generator_model_path = '../result_models/generator_weights.h5'
teeth_finder_model_path = '../result_models/teeth_finder_weights.h5'
numpys_path = '../patient_data/numpys/'
paths_pred_masks = '/*_image/*.png'

batch_size = 3
img_x, img_y = 512, 512


def load_patient():
    paths_predicts = natsorted(glob.glob(patient_path + paths_pred_masks))

    set_predictors = np.zeros((len(paths_predicts), img_x, img_y, 1), dtype=np.float16)

    for path_predictor, i in zip(paths_predicts, range(len(paths_predicts))):
        predictor = cv2.cvtColor(cv2.imread(path_predictor), cv2.COLOR_RGB2GRAY)
        set_predictors[i] = np.copy(predictor) / 255
        sys.stdout.write("\rImage %i loaded" % i)
    print('Patient loaded')
    return set_predictors


def clearer_load():
    model = keras.models.load_model(clearer_model_path)
    print('Clearer loaded')
    return model


def generator_load():
    model = keras.models.load_model(generator_model_path)
    print('Generator loaded')
    return model


def teeth_finder_load():
    model = keras.models.load_model(teeth_finder_model_path)
    print('Teeth finder loaded')
    return model


def xor_image_sets(orig, deleter):
    result = np.zeros((len(orig), img_x, img_y, 1), dtype=np.float16)
    for orig_image, deleter_image, iter in zip(orig, deleter, range(len(orig))):
        for i in range(img_x):
            for j in range(img_y):
                result[iter][i][j] = max(0, (orig_image[i][j] - deleter_image[i][j]))
    return result


def delete_teeth(orig, teeth):
    res = xor_image_sets(orig, teeth)
    print('Teeth deleted')
    return res


def membran_model_xor_creater(healthy_model_without_teeth, sick_mask):
    res = xor_image_sets(healthy_model_without_teeth, sick_mask)
    print('Model formed')
    return res


def clear_from_artifacts(membran):
    # Make by practice
    print('Cleared')


def save_npy(name, data):
    np.save(numpys_path + name, data)


raw_data = load_patient()

clearer_model = clearer_load()
generator_model = generator_load()
teeth_finder_model = teeth_finder_load()

cleared_data = clearer_model.predict(raw_data)
save_npy('cleared_masks.npy', cleared_data)
healthy_mask = generator_model.predict(cleared_data)
save_npy('healthy_masks.npy', healthy_mask)

sick_teeth_masks = teeth_finder_model.predict(cleared_data)
save_npy('sick_teeth_masks.npy', sick_teeth_masks)
healthy_teeth_masks = teeth_finder_model.predict(healthy_mask)
save_npy('healthy_teeth_masks.npy', healthy_teeth_masks)

sick_without_teeth = delete_teeth(cleared_data, sick_teeth_masks)
save_npy('sick_without_teeth.npy', sick_without_teeth)
healthy_without_teeth = delete_teeth(healthy_mask, healthy_teeth_masks)
save_npy('healthy_without_teeth.npy', healthy_without_teeth)

membran_model = membran_model_xor_creater(healthy_without_teeth, sick_without_teeth)
save_npy('membran_model.npy', membran_model)

cleared_membran_model = clear_from_artifacts(membran_model)
save_npy('result.npy', cleared_membran_model)
