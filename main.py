from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from Model import *

from PIL import Image
import os, sys
import shutil
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt


input_directory_hr_images = "img_align_celeba"
output_directory_lr_images = "img_align_celeba_lr"
output_directory_hr_images = "img_align_celeba_hr"


def clean_and_create_directory(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
    else:
        shutil.rmtree(directory)
        os.mkdir(directory)


def down_sampling(hr_directory):
    files = os.listdir(hr_directory)
    clean_and_create_directory(output_directory_hr_images)
    clean_and_create_directory(output_directory_lr_images)
    for file in files:
        img = Image.open(hr_directory + '/' + file)
        img_hr = img.resize((128, 128), Image.LANCZOS)
        img_lr = img.resize((64, 64), Image.LANCZOS)
        img_hr.save(output_directory_hr_images + '/' + file)
        img_lr.save(output_directory_lr_images + '/' + file)


# down_sampling(input_directory_hr_images)

def files_setup(input_directory, train_size):
    #   list all image files and then sorts them
    files = os.listdir(input_directory)
    files = sorted(files)
    files = files[:train_size]
    files = [os.path.join(input_directory, file) for file in files]
    return files


def image_read_decode(image):
    img = tf.io.read_file(image)
    img_decode = tf.image.decode_jpeg(img, channels=3)
    img_decode = tf.dtypes.cast(img_decode, tf.float32)
    img_norm = img_decode / 127.5
    img_norm = img_norm - 1
    return img_norm


def input_setup(batch_size):
    feature_files = files_setup(output_directory_lr_images, 10000)
    label_files = files_setup(output_directory_hr_images, 10000)

    #   Creates a queue based dataset of tensors

    feature_dataset = tf.data.Dataset.from_tensor_slices(feature_files)
    label_dataset = tf.data.Dataset.from_tensor_slices(label_files)

    #   Opens each files inside the dataset and decodes it

    feature_image = feature_dataset.map(image_read_decode, num_parallel_calls=multiprocessing.cpu_count())

    label_image = label_dataset.map(image_read_decode, num_parallel_calls=multiprocessing.cpu_count())

    dataset = tf.data.Dataset.zip((feature_image, label_image))
    batched_dataset = dataset.batch(batch_size)
    return batched_dataset


def un_trainable(model, layer_name):
    model.trainable = False
    for layer in model.layers:
        layer.trainable = False
    model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(name=layer_name).output)
    model.trainable = False
    return model


def plot_image(pred):
    pred = (pred+1)*127.5
    pred = tf.dtypes.cast(pred, tf.uint8)
    for image in pred:
        plt.imshow(image, interpolation='nearest')
        plt.show()
        break

def vgg_loss(y_true, y_pred):
    vgg19 = tf.keras.applications.vgg19.VGG19(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    model = un_trainable(vgg19, 'block5_conv4')
    return tf.keras.losses.mean_squared_error(model(y_true), model(y_pred))



def train(batch_size, epochs):
    generator_model = generator((64, 64, 3))
    discriminator_model = discriminator((128, 128, 3))

    generator_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1E-4, beta_1=0.9,
                                                               beta_2=0.999, epsilon=1e-08), loss=vgg_loss)

    discriminator_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1E-4, beta_1=0.9,
                                                               beta_2=0.999, epsilon=1e-08),
                                loss='binary_crossentropy')

    discriminator_model.trainable = False

    model_input = tf.keras.layers.Input((64, 64, 3))
    model = generator_model(model_input)
    model_output = discriminator_model(model)
    model = tf.keras.Model(inputs=model_input, outputs=[model, model_output])
    model.compile(loss=[vgg_loss, 'binary_crossentropy'], loss_weights=[1., 1e-3], optimizer=tf.keras.optimizers.Adam(
        learning_rate=1E-4, beta_1=0.9,
        beta_2=0.999, epsilon=1e-08
    ))

    clean_and_create_directory('saved_model')
    train_dataset = input_setup(batch_size)
    for epoch in range(epochs):
        batch_count = 0
        for (lr_image, hr_image) in train_dataset:
            generator_predicted = generator_model.predict(lr_image)


            real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.2
            fake_data_Y = np.random.random_sample(batch_size) * 0.2

            discriminator_model.trainable = True
            d_loss_real = discriminator_model.train_on_batch(hr_image, real_data_Y)
            d_loss_fake = discriminator_model.train_on_batch(generator_predicted, fake_data_Y)
            discriminator_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

            gan_Y = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.2
            discriminator_model.trainable = False
            gan_loss = model.train_on_batch(lr_image, [hr_image, gan_Y])
            batch_count += 1
            print(batch_count)
        print("discriminator_loss : %f" % discriminator_loss)
        print("gan_loss :", gan_loss)
        # print(plot_image(generator_model.predict(lr_image)))

        plot_image(generator_model.predict(lr_image))
        generator_model.save('saved_model/'+'generator_model%d.h5' % epoch)
