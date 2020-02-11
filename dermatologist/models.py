"""
Module Docstring
"""

__author__ = "Justin Solms"
__version__ = "0.0.0"
__license__ = "GPL"

import numpy as np
import pandas as pd
import pickle
import os

from datetime import datetime

from math import ceil

import argparse
from logzero import logger

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from keras.applications.inception_v3 import InceptionV3
# from keras.applications.xception import Xception
# from keras.applications.vgg16 import VGG16
# from keras.applications.vgg19 import VGG19
# from keras.applications.resnet50 import ResNet50
# from keras.applications.mobilenet import MobileNet

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense, Input
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import CSVLogger
from keras.applications import VGG16, Xception
from keras.models import Model
from keras.optimizers import Adam

from dermatologist.data import Data


class Model(object):

    output_path = 'output'
    log_path = 'log.{}.csv'
    history_path = 'history.{}.pkl'
    best_model_path = 'weights.best.{}.hdf5'
    data_dir = os.path.join('dermatologist', 'data')

    def __init__(self,
                 epochs=100,
                 batch_size=48,
                 n_dense=512,
                 dropout=0.2,
                 learn_rate=0.0002,
                 n_outputs=7,
                 target_size=(192, 192),
                 validation_split=0.2,
                 steps_per_epoch=None,
                 ):

        self.dropout = dropout
        self.n_dense = n_dense
        self.n_outputs = n_outputs
        self.epochs = epochs
        self.batch_size = batch_size
        self.learn_rate = learn_rate
        self.target_size = target_size
        self.validation_split = validation_split
        self.steps_per_epoch = steps_per_epoch

        # Instantiate data augmentation object
        self.data = Data(
            validation_split=self.validation_split,
            batch_size=self.batch_size,
            target_size=self.target_size,
            )

        #  Make model save directory
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

        #  Prefix (or suffix or in-between) for training job outputs batch
        self.identifier = '{}-e:{}-b:{}-n:{}-d:{}-l:{}'.format(
            datetime.now().strftime('%FT%T'),
            epochs, batch_size, n_dense, dropout, learn_rate,
        )

        # Best weights path.
        self.best_model_path = os.path.join(
            self.output_path, self.best_model_path.format(self.identifier) )

        # Log path.
        self.log_path = os.path.join(
            self.output_path, self.log_path.format(self.identifier) )

        # Save history object
        self.history_path = os.path.join(
            self.output_path, self.history_path.format(self.identifier) )

        # ImageNet InceptionV3 base network.
        input_tensor = Input(shape=self.data.shape)
        # Base network model with fixed weights
        logger.info('Initializing base model.')
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_tensor=input_tensor,
            pooling=None,  # So we fully control the top model interface
            )
        self.base_model = base_model

        # Top network model added to base model
        logger.info('Initializing top model.')
        model = Sequential()
        model.add(self.base_model)
        model.add(GlobalAveragePooling2D(
            input_shape=base_model.output_shape[1:]))
        model.add(Dense(n_dense, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(n_dense, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(n_outputs, activation='softmax'))

        #  Set up model training
        logger.info('Compiling  model.')
        model.compile(
            optimizer=Adam(lr=self.learn_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'categorical_crossentropy'],
            )
        self.model = model

        # Initialize base layers to non-trainable
        # FIXME: Log sum more!
        self.set_base_trainable_layers()

        logger.info('Creating callbacks')

        # Callback for early stropping
        self.stopper = EarlyStopping(
            monitor='val_loss', min_delta=0,
            patience=20, verbose=1, mode='auto')

        # Callback to save best weights
        self.checkpointer = ModelCheckpoint(
            filepath=self.best_model_path,
            verbose=1,
            save_best_only=True,
            save_weights_only=True)

        # Callback to save dat logs to CSV
        self.csv_logger = CSVLogger(self.log_path)

    def set_base_trainable_layers(self, layer_names=None):
        # Get all layer names
        model_layer_names = [layer.name for layer in self.base_model.layers]

        if layer_names is not None:
            # Check if all specified layer_names are in model layer names
            if not all(name in model_layer_names for name in layer_names):
                raise ValueError('Unexpected layer_names, not in base model.')
            for layer in self.base_model.layers:
                if layer.name in layer_names:
                    layer.trainable = True
                else:
                    layer.trainable = False
            # We have set layers, now set model
            self.base_model.trainable = True
        else:
            #  Set entire base mode not trainable
            for layer in self.base_model.layers:
                layer.trainable = False
            # We have set layers, now set model
            self.base_model.trainable = False

        # Print layers training table
        # Source - Dipanjan (DJ) Sarkar
        layers = [(layer, layer.name, layer.trainable)
                  for layer in self.base_model.layers]
        output = pd.DataFrame(
            layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
        print(output)

    def fit(self):
        # Set steps per epoch
        if self.steps_per_epoch is None:
            steps_per_epoch = ceil(self.data.num_samples / self.batch_size)
        else:
            steps_per_epoch = self.steps_per_epoch

        # Train the model
        logger.info('Training steps per epoch {}.'.format(steps_per_epoch))
        self.history = self.model.fit_generator(
            generator=self.data.train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=self.epochs,
            callbacks=[
                self.checkpointer,
                self.stopper,
                self.csv_logger,
                ],
            workers=2,
            use_multiprocessing=True,
            verbose=2)

        #  Load best model weights
        path = os.path.join(self.output_path, self.best_model_path)
        if os.path.exists(path):
            self.model.load_weights(path)

    def predict(self):
        pass

    def report(self):
        #  Make predictions from top test features generated by the base from
        #  the test data.
        predictions = list()
        for feature in self.test_features:
            predictions.append(
                self.model.predict(np.expand_dims(feature, axis=0))
                )
        predictions = np.array(predictions).squeeze()

        # Prepare output format for sklearn reporting
        categories = self.data.categories
        actual_n = np.argmax(self.data.test_labels, axis=1)
        actual = categories[actual_n]
        predicted_n = np.argmax(predictions, axis=1)
        predicted = categories[predicted_n]

        print()
        print ('Confusion Matrix :\n', confusion_matrix(actual, predicted))
        print()
        print ('Accuracy Score :\n', accuracy_score(actual, predicted))
        print()
        print ('Report : \n', classification_report(actual, predicted))
        print()

    def print_summary(self):
        print(self.base_model.summary())
        print(self.model.summary())