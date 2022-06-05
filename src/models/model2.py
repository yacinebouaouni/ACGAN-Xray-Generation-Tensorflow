#! /usr/bin/python3

from __future__ import print_function, division
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, concatenate, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, Embedding
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu, tanh, linear
from tensorflow import keras
from tensorflow.keras.utils import Progbar
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from collections import defaultdict
import pickle as pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score


def centralize(image):
    return (image - 127.5) / 127.5


class ACGAN():

    def __init__(self, data_path, latent_dim, batch_size=64):

        self.img_rows = 112
        self.img_cols = 112
        self.channels = 3
        self.ims = (self.img_rows, self.img_cols)
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 2
        self.latent_dim = latent_dim
        self.variance = 0.02
        self.valid_bs = 16
        self.batch_size = batch_size

        # Data generator:
        train_datagen = ImageDataGenerator(preprocessing_function=centralize)
        self.train_generator = train_datagen.flow_from_directory(
            data_path + 'train',
            target_size=self.ims,
            batch_size=self.batch_size,
            class_mode='binary', shuffle=True)

        test_datagen = ImageDataGenerator(preprocessing_function=centralize)
        self.validation_generator = test_datagen.flow_from_directory(
            data_path + 'test',
            target_size=self.ims,
            batch_size=16,
            class_mode='binary', shuffle=True)


        self.optimizer = Adam(0.0002, 0.5)
        self.loss = ['binary_crossentropy', 'sparse_categorical_crossentropy']

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        self.generator = self.build_generator()

        self.gan = self.define_gan(self.generator, self.discriminator)

        print(self.gan.summary())

        if not os.path.exists('saved_models/'):
            os.mkdir('saved_models/')

        if not os.path.exists('saved_images/'):
            os.mkdir('saved_images/')

    def build_generator(self):

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')

        noise_branch = Dense(1024 * 7 * 7)(noise)
        noise_branch = relu(noise_branch)
        noise_branch = Reshape((7, 7, 1024))(noise_branch)
        noise_branch = Model(inputs=noise, outputs=noise_branch)

        label_branch = Embedding(input_dim=50, output_dim=1)(label)
        label_branch = Dense(49, input_shape=(7, 7))(label_branch)
        label_branch = linear(label_branch)
        label_branch = Reshape((7, 7, 1), )(label_branch)
        label_branch = Model(inputs=label, outputs=label_branch)
        combined = concatenate([noise_branch.output, label_branch.output])

        combined = Conv2DTranspose(512, (5, 5), strides=(2, 2), padding="same")(combined)
        combined = BatchNormalization(momentum=0)(combined)
        combined = relu(combined)

        combined = Conv2DTranspose(256, (5, 5), strides=(2, 2), padding="same")(combined)
        combined = BatchNormalization(momentum=0)(combined)
        combined = relu(combined)

        combined = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding="same")(combined)
        combined = BatchNormalization(momentum=0)(combined)
        combined = relu(combined)

        combined = Conv2DTranspose(self.channels, (5, 5), strides=(2, 2), padding="same")(combined)
        combined = tanh(combined)

        model = Model(inputs=[label_branch.input, noise_branch.input], outputs=combined)

        keras.utils.plot_model(model, "generateur.png", show_shapes=True)

        return model

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), input_shape=self.img_shape, padding="same"))
        model.add(BatchNormalization(momentum=0))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))

        model.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2), input_shape=self.img_shape, padding="same"))
        model.add(BatchNormalization(momentum=0))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))

        model.add(Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding="same"))
        model.add(BatchNormalization(momentum=0))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))

        model.add(Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding="same"))
        model.add(BatchNormalization(momentum=0))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))

        model.add(Conv2D(512, kernel_size=(3, 3), strides=(2, 2), padding="same"))
        model.add(BatchNormalization(momentum=0))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))
        model.add(Flatten())

        img = Input(shape=self.img_shape)
        features = model(img)

        validity = Dense(1, activation="sigmoid")(features)
        label = Dense(self.num_classes, activation="softmax")(features)

        keras.utils.plot_model(model, "discriminateur.png", show_shapes=True)

        return Model(img, [validity, label])

    # define the combined generator and discriminator model, for updating the generator
    def define_gan(self, g_model, d_model):
        # make weights in the discriminator not trainable
        d_model.trainable = False

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([label, noise])

        valid, target_label = self.discriminator(img)

        model = Model([label, noise], [valid, target_label])
        # compile model
        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model



    def load_real(self, training):
        if training:
            return self.train_generator.next()
        else:
            return self.validation_generator.next()

    def generate_fake(self, batch_size):

        noise = np.random.normal(0, self.variance, (batch_size, self.latent_dim))
        sampled_labels = np.random.randint(0, 1, batch_size)

        fake_images = self.generator.predict([sampled_labels.reshape((-1, 1)), noise], verbose=0)

        return fake_images, sampled_labels

    # evaluate the discriminator, plot generated images, save generator model
    def summarize_performance(self, epoch):
        # prepare real samples
        X_real, y_real = self.load_real(training=False)
        bsv = X_real.shape[0]
        # prepare fake examples
        X_fake, y_fake = self.generate_fake(bsv)

        X = np.concatenate((X_real, X_fake))
        y = np.array([1] * bsv + [0] * bsv)
        aux_y = np.concatenate((y_real, y_fake), axis=0)

        discriminator_test_loss = self.discriminator.evaluate(X, [y, aux_y], verbose=False)

        accuracy = 1-np.mean(np.array(self.discriminator.predict(X_fake)[0]))

        return discriminator_test_loss, accuracy



    def eval_gen(self):

        noise = np.random.normal(0, self.variance, (2 * self.valid_bs, self.latent_dim))

        sampled_labels = np.random.randint(0, 1, 2 * self.valid_bs)

        trick = np.ones(2 * self.valid_bs)

        generator_test_loss = self.gan.evaluate([sampled_labels.reshape((-1, 1)), noise],[trick, sampled_labels], verbose=False)

        return generator_test_loss




    def train(self, epochs, save_every):


        train_history = defaultdict(list)
        test_history = defaultdict(list)

        for epoch in range(epochs):

            print('Epoch {} of {}'.format(epoch + 1, epochs))
            nb_batches = len(self.train_generator)
            progress_bar = Progbar(target=nb_batches)

            epoch_gen_loss = []
            epoch_disc_loss = []

            for index in range(nb_batches):


                progress_bar.update(index)


                image_batch, label_batch = self.load_real(training=True)
                bs = label_batch.shape[0]
                generated_images, sampled_labels = self.generate_fake(bs)
                X = np.concatenate((image_batch, generated_images))
                y = np.array([1] * bs + [0] * bs)
                aux_y = np.concatenate((label_batch, sampled_labels))
                epoch_disc_loss.append(self.discriminator.train_on_batch(X, [y, aux_y]))


                # Train the GAN and update generator with the discriminator error.
                X_gan = np.random.normal(0, self.variance, (self.batch_size, self.latent_dim))
                labels_gan = np.random.randint(0, 1, self.batch_size)
                y_gan = np.ones(self.batch_size)
                epoch_gen_loss.append(self.gan.train_on_batch([labels_gan.reshape((-1, 1)), X_gan], [y_gan, labels_gan]))


            if epoch % 10 == 0:
                #print('\nTesting for epoch {}:'.format(epoch + 1))
                discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)
                discriminator_test_loss, acc = self.summarize_performance(epoch)
                generator_test_loss = self.eval_gen()
                generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)


                train_history['generator'].append(generator_train_loss)
                train_history['discriminator'].append(discriminator_train_loss)
                test_history['generator'].append(generator_test_loss)
                test_history['discriminator'].append(discriminator_test_loss)
                test_history['Accuracy'].append(acc)


                #ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<4.2f} | {3:<15.2f}'
                #ACC_FMT = '{0:<4.2f}'
                #print(ROW_FMT.format('generator (train)', *train_history['generator'][-1]))
                #print(ROW_FMT.format('generator (test)', *test_history['generator'][-1]))
                #print(ROW_FMT.format('discriminator (train)', *train_history['discriminator'][-1]))
                #print(ROW_FMT.format('discriminator (test)', *test_history['discriminator'][-1]))
                #print('Accuracy {:.2f}'.format(100 * acc))
                pickle.dump({'train': train_history, 'test': test_history}, open('acgan-history.pkl', 'wb'))

            if epoch % save_every == 0:


                self.generator.save('saved_models/generator.hdf5')
                self.discriminator.save('saved_models/discriminator.hdf5')


            if epoch % 10 == 0:
                r, c = 1, 4
                fig, axs = plt.subplots(r, c)

                for i in range(c):
                    noise = np.random.normal(0, self.variance, (1, self.latent_dim))
                    sampled_labels = np.random.randint(0, 1, 1)
                    gen_imgs = self.generator.predict([sampled_labels, noise])
                    gen_imgs = 127 * gen_imgs + 127
                    axs[i].imshow(gen_imgs[0, :, :, 0], cmap='gray')

                fig.savefig("saved_images/%d.png" % epoch)
                plt.close()
