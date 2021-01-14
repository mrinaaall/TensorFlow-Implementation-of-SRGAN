import tensorflow as tf


def SubpixelConv2D(input_shape,name,scale=4):
    def subpixel_shape(input_shape):
        dims = [input_shape[0],
                input_shape[1] * scale,
                input_shape[2] * scale,
                int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        return tf.nn.depth_to_space(x, scale)

    return tf.keras.layers.Lambda(subpixel, output_shape=subpixel_shape, name=name)


def identity_block(model, blocks_number):
    kernel_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None)
    gamma_init = tf.keras.initializers.RandomNormal(mean=1.0, stddev=0.02, seed=None)
    model_shortcut = model
    for i in range(blocks_number):
        model = tf.keras.layers.Conv2D(64, (3, 3), (1, 1), padding='same', kernel_initializer=kernel_init
                                       , bias_initializer=None)(model)
        model = tf.keras.layers.BatchNormalization(axis=-1, gamma_initializer=gamma_init, momentum=0.5)(model)
        model = tf.keras.layers.PReLU(shared_axes=[1, 2])(model)
        model = tf.keras.layers.Conv2D(64, (3, 3), (1, 1), padding='same', kernel_initializer=kernel_init
                                       , bias_initializer=None)(model)
        model = tf.keras.layers.BatchNormalization(axis=-1, gamma_initializer=gamma_init, momentum=0.5)(model)
        model = tf.keras.layers.Add()([model_shortcut, model])
    return model


def generator(input_shape):
    kernel_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None)
    gamma_init = tf.keras.initializers.RandomNormal(mean=1.0, stddev=0.02, seed=None)

    model_input = tf.keras.layers.Input(input_shape)
    model = tf.keras.layers.Conv2D(64, (9, 9), (1, 1), padding='same',
                                   kernel_initializer=kernel_init)(model_input)
    model = tf.keras.layers.PReLU(shared_axes=[1, 2])(model)

    model_shortcut = model

    model = identity_block(model, 16)

    model = tf.keras.layers.Conv2D(64, (3, 3), (1, 1), padding='same', kernel_initializer=kernel_init
                                   , bias_initializer=None)(model)
    model = tf.keras.layers.BatchNormalization(axis=-1, gamma_initializer=gamma_init, momentum=0.5)(model)
    model = tf.keras.layers.Add()([model_shortcut, model])

    model = tf.keras.layers.Conv2D(256, (3, 3), (1, 1), padding='same',
                                   kernel_initializer=kernel_init, bias_initializer=None)(model)
    model = SubpixelConv2D(model.get_shape(), 'subpixel1', scale=2)(model)
    model = tf.keras.layers.PReLU(shared_axes=[1, 2])(model)

    # model = tf.keras.layers.Conv2D(256, (3, 3), (1, 1), padding='same',
    #                                kernel_initializer=kernel_init, bias_initializer=None)(model)
    # model = SubpixelConv2D(model.get_shape(), 'subpixel2', scale=2)(model)
    # model = tf.keras.layers.Activation('relu')(model)

    model = tf.keras.layers.Conv2D(3, (9, 9), (1, 1), padding='same', kernel_initializer=kernel_init,
                                   bias_initializer=None)(model)

    model = tf.keras.layers.Activation('tanh')(model)

    model = tf.keras.Model(inputs=model_input, outputs=model, name='generator')
    return model


def discriminator_block(model, kernel_init, gamma_init, factor, stride, initial_filter=64, norm=True):
    if norm:
        model = tf.keras.layers.Conv2D(initial_filter * factor, (3, 3), stride, padding='same',
                                       kernel_initializer=kernel_init,
                                       bias_initializer=None)(model)
        model = tf.keras.layers.BatchNormalization(axis=-1, gamma_initializer=gamma_init, momentum=0.5)(model)
        model = tf.keras.layers.LeakyReLU(alpha=0.2)(model)

    else:
        model = tf.keras.layers.Conv2D(initial_filter * factor, (3, 3), stride, padding='same',
                                       kernel_initializer=kernel_init,
                                       bias_initializer=None)(model)
        model = tf.keras.layers.LeakyReLU(alpha=0.2)(model)

    return model


def discriminator(input_shape):
    kernel_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None)
    gamma_init = tf.keras.initializers.RandomNormal(mean=1.0, stddev=0.02, seed=None)

    model_input = tf.keras.layers.Input(input_shape)

    model = discriminator_block(model_input, kernel_init, gamma_init, 1, (1, 1), 64, False)
    model = discriminator_block(model, kernel_init, gamma_init, 1, (2, 2), 64, True)
    model = discriminator_block(model, kernel_init, gamma_init, 2, (1, 1), 64, True)
    model = discriminator_block(model, kernel_init, gamma_init, 2, (2, 2), 64, True)
    model = discriminator_block(model, kernel_init, gamma_init, 4, (1, 1), 64, True)
    model = discriminator_block(model, kernel_init, gamma_init, 4, (2, 2), 64, True)
    model = discriminator_block(model, kernel_init, gamma_init, 8, (1, 1), 64, True)
    model = discriminator_block(model, kernel_init, gamma_init, 8, (2, 2), 64, True)

    model = tf.keras.layers.Flatten()(model)
    model = tf.keras.layers.Dense(1024)(model)
    model = tf.keras.layers.LeakyReLU(alpha=0.2)(model)

    model = tf.keras.layers.Dense(1)(model)
    model = tf.keras.layers.Activation('sigmoid')(model)

    model = tf.keras.Model(inputs=model_input, outputs=model, name='discriminator')

    return model
