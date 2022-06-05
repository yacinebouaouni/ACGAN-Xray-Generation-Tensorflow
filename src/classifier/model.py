from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from livelossplot.inputs.keras import PlotLossesCallback
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from sklearn.metrics import accuracy_score



class model:

    def __init__(self, path):

        print(path)
        self.path = path  # path to data
        self.width = 112
        self.height = 112
        self.channel = 3

        self.imshape = (self.height, self.width, self.channel)

        self.batch_size = 124
        self.valid_batch_size = 200

        self.optimizer = Adam(learning_rate=0.0001)
        self.model = self.build_model()


        self.train_gen, self.val_gen, self.test_gen = self.build_data()

    def build_model(self):

        model = VGG16(include_top=False, input_shape=(112, 112, 3))
        model.trainable = True

        flat = Flatten()(model.layers[-1].output)
        dense1 = Dense(16, activation='relu')(flat)
        dense1 = Dropout(0.5)(dense1)
        output = Dense(1, activation='sigmoid')(dense1)

        model = Model(inputs=model.inputs, outputs=output)

        model.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def build_data(self):


        print('Loading data from ' + self.path )
        train_generator = ImageDataGenerator(rescale=1.0/255.0) # VGG16 preprocessing

        traingen = train_generator.flow_from_directory(self.path+'train/',
                                                       target_size=(self.height, self.width),
                                                       class_mode='binary',
                                                       batch_size=self.batch_size,
                                                       shuffle=True,
                                                       seed=42)

        validgen = train_generator.flow_from_directory(self.path+'test/',
                                                       target_size=(self.height, self.width),
                                                       class_mode='binary',
                                                       batch_size=self.valid_batch_size,
                                                       shuffle=True,
                                                       seed=42)

        testgen = train_generator.flow_from_directory(self.path+'test/',
                                                       target_size=(self.height, self.width),
                                                       class_mode='binary',
                                                       batch_size=1,
                                                       shuffle=False,
                                                       seed=42)

        return traingen, validgen, testgen

    def train(self, epochs):

        n_steps = self.train_gen.samples // self.batch_size
        n_val_steps = self.val_gen.samples // self.valid_batch_size

        plot_loss_callback = PlotLossesCallback()

        # ModelCheckpoint callback - save best weights
        checkpoint_callback = ModelCheckpoint(filepath='tl_model_v1.weights.best.hdf5',
                                          save_best_only=True,
                                          verbose=1)

        # EarlyStopping
        early_stop_callback = EarlyStopping(monitor='val_loss',
                                   patience=10,
                                   restore_best_weights=True,
                                   mode='min')

        vgg_history = self.model.fit(self.train_gen,
                                    batch_size=self.batch_size,
                                    epochs=epochs,
                                    validation_data=self.val_gen,
                                    steps_per_epoch=n_steps,
                                    validation_steps=n_val_steps,
                                    callbacks=[checkpoint_callback, early_stop_callback, plot_loss_callback],
                                    verbose=1)



    def test(self):

        self.model.load_weights('tl_model_v1.weights.best.hdf5')
        preds = self.model.predict(self.test_gen)
        preds = np.round(preds).astype(int)

        labels = self.test_gen.classes

        vgg_acc = accuracy_score(labels, preds)
        print("VGG16 Model Accuracy without Fine-Tuning: {:.2f}%".format(vgg_acc * 100))

