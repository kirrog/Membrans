from tensorflow import keras

model_path = '..\\models\\generator_weights.h5'




def load_model():
    new_model = keras.models.load_model(model_path)
    return new_model


def save_generator_model(model):
    model.save(model_path)

