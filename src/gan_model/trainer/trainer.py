import os
import sys
import time

import tensorflow as tf
from datetime import datetime

from tqdm import tqdm

from src.gan_model.dataset_generator_provider.dataset_generator_providers import generator_dataset_pair_creater
from src.utils.config_loader import train_data
from src.utils.config_loader import test_data
from src.clearer.dataset_generators.dataset_generator_providers import clearer_dataset_pair_creater

log_dir = '../models/generator/logs/'
checkpoint_prefix = os.path.join(log_dir, "ckpt")
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# mse_entropy = tf.keras.losses.MSE()


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def train_generator_model(model_coder, model_decoder, model_discriminator):
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     coder=model_coder,
                                     decoder=model_decoder,
                                     discriminator=model_discriminator)
    # checkpoint.restore(checkpoint_prefix + "checkpoint")

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=1, monitor='loss', mode='min', min_delta=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}_' + datetime.now().strftime(
                "%Y%m%d-%H%M%S") + '.h5',
            monitor='loss', mode='min')
    ]
    # -loss{loss:.3f}-val_loss{val_loss:.3f}_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '

    batch_size = 3
    epochs = 50
    noise_dim = 100
    num_examples_to_generate = 16

    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    @tf.function
    def train_step(images, com):

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            coded_images = model_coder(images[0], training=True)
            generated_images = model_decoder(coded_images, training=True)
            fake_output = model_discriminator(generated_images, training=True)
            real_output = model_discriminator(images[1], training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_decoder = gen_tape.gradient(gen_loss, model_decoder.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, model_discriminator.trainable_variables)

        if com == 0:
            generator_optimizer.apply_gradients(zip(gradients_of_decoder, model_decoder.trainable_variables))
            discriminator_optimizer.apply_gradients(
                zip(gradients_of_discriminator, model_discriminator.trainable_variables))
        elif com == -1:
            discriminator_optimizer.apply_gradients(
                zip(gradients_of_discriminator, model_discriminator.trainable_variables))
        else:
            generator_optimizer.apply_gradients(zip(gradients_of_decoder, model_decoder.trainable_variables))

        return gen_loss, disc_loss

    @tf.function
    def test_step(images):
        coded_images = model_coder(images[0], training=False)
        generated_images = model_decoder(coded_images, training=False)
        fake_output = model_discriminator(generated_images, training=False)

        return generator_loss(fake_output)

    train_dataset = generator_dataset_pair_creater(train_data)
    test_dataset = generator_dataset_pair_creater(test_data)

    def train(dataset_train, dataset_test, epochs):
        for epoch in range(epochs):
            start = time.time()
            count = 0
            gen_loss_accum = 0.0
            dis_loss_accum = 0.0
            com = 0

            for image_batch in dataset_train:
                gen_loss, disc_loss = train_step(image_batch, com)
                gen_loss_accum += float(gen_loss.numpy())
                dis_loss_accum += float(disc_loss.numpy())
                count += 1
                gen_loss_ = gen_loss_accum / count
                dis_loss_ = dis_loss_accum / count
                if dis_loss_ < 1.0 and gen_loss_ < 0.9:
                    com = 0
                elif dis_loss_ < 1.0:
                    com = 1
                else:
                    com = -1

                sys.stdout.write(
                    f"\rTrain {epoch + 1:02d} is {time.time() - start:0.1f} sec | com {com:01d} | gen_loss {gen_loss_accum / count:0.5f} dis_loss {dis_loss_accum / count:0.5f}")
            print()
            start = time.time()
            test_loss = 0.0
            count_checks = 0
            count = 0
            for image_batch in dataset_test:
                res = test_step(image_batch)
                k = float(res.numpy())
                test_loss += abs(k)
                if abs(k) < 0.5:
                    count_checks += 1
                count += 1
                sys.stdout.write(
                    f"\rTest {epoch + 1:02d} is {time.time() - start:0.1f} loss {test_loss / count:0.5f} acc {count_checks / count:0.5f}")
            print()
            if (epoch + 1) % 1 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)

    train(train_dataset, test_dataset, epochs)


def test_generator_model(model_coder, model_decoder, model_discriminator):
    epochs = 50

    @tf.function
    def test_step(images):
        coded_images = model_coder(images[0], training=False)
        generated_images = model_decoder(coded_images, training=False)
        fake_output = model_discriminator(generated_images, training=False)

        return generator_loss(fake_output)

    test_dataset = generator_dataset_pair_creater(test_data)
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    for i in range(epochs):

        checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                         discriminator_optimizer=discriminator_optimizer,
                                         coder=model_coder,
                                         decoder=model_decoder,
                                         discriminator=model_discriminator)
        checkpoint.restore(f"/home/kirrog/Documents/projects/Membrans/models/generator/logs/ckpt-{i + 1}.index")
        start = time.time()
        test_loss = 0.0
        count_checks = 0
        count = 0
        for image_batch in test_dataset:
            res = test_step(image_batch)
            k = float(res.numpy())
            test_loss += abs(k)
            if abs(k) < 0.5:
                count_checks += 1
            count += 1
            sys.stdout.write(
                f"\rTest {i + 1:02d} is {time.time() - start:0.1f} loss {test_loss / count:0.5f} acc {count_checks / count:0.5f}")
        print()
