import glob
import os
import sys

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from natsort import natsorted

from src.gan_model.model.generator_model_u_net import generator_model_decoder, generator_model_coder, \
    generator_model_discriminator
from src.utils.config_loader import one_test_data_dir
from src.utils.config_loader import output_data_dir

log_dir = '../models/generator/logs/'
checkpoint_prefix = os.path.join(log_dir, "ckpt")

img_x, img_y = 512, 512


def rgb2green(image):
    return np.float32(np.multiply(image[:, :, 1], image[:, :, 2]))


def read_from_png():
    paths_predicts = natsorted(glob.glob(one_test_data_dir + '/ORIG/*.png'))
    print(len(paths_predicts))
    pred = []
    for file, iter in zip(paths_predicts, range(len(paths_predicts))):
        image_2d = rgb2green(plt.imread(file))
        image_2d = (np.maximum(image_2d, 0) / image_2d.max())
        image_2d = cv2.resize(image_2d, (img_x, img_y), interpolation=cv2.INTER_CUBIC)
        image_2d = image_2d.reshape((1, img_x, img_y, 1))
        pred.append(image_2d)
        sys.stdout.write("\rImage %i loaded" % iter)
    print()
    return pred


def test_generator_model(model_coder, model_decoder, model_discriminator):
    checkp = 6

    def test_step(images):
        coded_images = model_coder(images, training=False)
        minimum = 1
        output = tf.constant(0)
        for i in range(200):
            generated_images = model_decoder(coded_images, training=False)
            discrim = model_discriminator(generated_images)

            discrim = abs(float(discrim))
            if discrim < 0.01:
                return generated_images, discrim
            else:
                if minimum > discrim:
                    minimum = discrim
                    output = generated_images

        return output, minimum

    val_dataset = read_from_png()

    checkpoint = tf.train.Checkpoint(coder=model_coder,
                                     decoder=model_decoder,
                                     discriminator=model_discriminator)
    checkpoint.restore(f"/home/kirrog/Documents/projects/Membrans/models/generator/logs/ckpt-{checkp}.index")

    count = 0
    # for image_batch in tqdm(val_dataset, desc="progress"):
    for image_batch in val_dataset:
        res, discrim = test_step(image_batch)
        res = res.numpy()
        d = np.zeros((res.shape[1], res.shape[2], 3))
        d[:, :, 1] = np.reshape(res, (res.shape[1], res.shape[2]))
        plt.imsave(output_data_dir + f"/{count:04d}.png", d)
        sys.stdout.write(f"\rImage {count:05d} written discrim {discrim:0.5f}")
        count += 1

    print()


model_co = generator_model_coder()
model_de = generator_model_decoder()
model_di = generator_model_discriminator()
test_generator_model(model_co, model_de, model_di)
