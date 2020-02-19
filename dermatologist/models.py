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
from keras.callbacks import CSVLogger, ProgbarLogger
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
                 target_size=(192, 192),
                 test_split=0.2,
                 steps_per_epoch=None,
                 ):

        self.dropout = dropout
        self.n_dense = n_dense
        self.epochs = epochs
        self.batch_size = batch_size
        self.learn_rate = learn_rate
        self.target_size = target_size
        self.test_split = test_split
        self.steps_per_epoch = steps_per_epoch


        # Instantiate data augmentation object
        self.data = Data(
            test_split=self.test_split,
            batch_size=self.batch_size,
            target_size=self.target_size,
            )
        self.n_outputs = self.data.n_classes
        n_outputs = self.data.n_classes

        #  Make model save directory
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

        #  Prefix (or suffix or in-between) for training job outputs batch
        self.identifier = '{}-e:{}-b:{}-n:{}-d:{}-l:{}-s:{}'.format(
            datetime.now().strftime('%FT%T'),
            epochs, batch_size, n_dense, dropout, learn_rate, steps_per_epoch
        )
        logger.info('Train Id is: {}'.format(self.identifier))

        # Best weights path.
        self.best_model_path = os.path.join(
            self.output_path, self.best_model_path.format(self.identifier) )

        # Log path.
        self.log_path = os.path.join(
            self.output_path, self.log_path.format(self.identifier) )

        # Save history object
        self.history_path = os.path.join(
            self.output_path, self.history_path.format(self.identifier) )


        self.add_base_model()

        self.add_top_model()

        self.set_base_trainable_layers()

        logger.info('Creating callbacks')

        # Callback for early stropping
        self.stopper = EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=5,
            mode='auto',
            verbose=1,
            )

        # Callback to save best weights
        self.checkpointer = ModelCheckpoint(
            monitor='val_loss',
            filepath=self.best_model_path,
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
            )

        # Callback to save dat logs to CSV
        self.csv_logger = CSVLogger(self.log_path)

    def add_base_model(self):
        logger.info('Adding base model.')
        # ImageNet InceptionV3 base network.
        input_tensor = Input(shape=self.data.shape)
        # Base network model with fixed weights
        self.base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_tensor=input_tensor,
            pooling=None,  # So we fully control the top model interface
            )

    def add_top_model(self):
        # Top network model added to base model
        logger.info('Adding top model - Interface shape: {}'.format(
            self.base_model.output_shape[1:]))
        self.model = Sequential()
        self.model.add(self.base_model)
        self.model.add(GlobalAveragePooling2D(
            input_shape=self.base_model.output_shape[1:]))
        self.model.add(Dense(self.n_dense, activation='relu'))
        self.model.add(Dropout(self.dropout))
        # self.model.add(Dense(self.n_dense, activation='relu'))
        # self.model.add(Dropout(self.dropout))
        self.model.add(Dense(self.n_outputs, activation='softmax'))

    def compile_model(self):
        logger.info('Compiling  model.')
        self.model.compile(
            optimizer=Adam(lr=self.learn_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'categorical_crossentropy'],
            )

    def set_base_trainable_layers(self, layer_names=None):
        logger.info('Setting trainable layers.')
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

        # It is necessary to compile model again after these changes.
        self.compile_model()

        # Print layers training table
        # Source - Dipanjan (DJ) Sarkar
        layers = [(layer, layer.name, layer.trainable)
                  for layer in self.base_model.layers]
        output = pd.DataFrame(
            layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
        print(output)

    def fit(self):
        logger.info('Fitting model.')
        data = self.data  # Training data object
        # Set steps per epoch
        if self.steps_per_epoch is None or self.steps_per_epoch == 0:
            # Keep the steps per epoch and validation steps proportionate.
            steps_per_epoch = data.train_flow.samples // self.batch_size
            validation_steps = data.validation_flow.samples // self.batch_size
        else:
            steps_per_epoch = self.steps_per_epoch
            validation_steps = self.steps_per_epoch * data.test_split

        # Train the model
        logger.info('Training steps per epoch: {}.'.format(steps_per_epoch))
        self.history = self.model.fit_generator(
            generator=data.train_flow,
            validation_data=data.validation_flow,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            epochs=self.epochs,
            callbacks=[
                self.checkpointer,
                self.stopper,
                self.csv_logger,
                ],
            use_multiprocessing = True,
            verbose=1,
            )

        #  Load best model weights
        path = os.path.join(self.output_path, self.best_model_path)
        if os.path.exists(path):
            self.model.load_weights(path)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def test_predictions(self, steps=None):
        predictions = self.model.predict_generator(
            self.data.test_flow,
            steps=steps,
            use_multiprocessing=True,
            verbose=1,
        )
        return predictions

    def report(self, test_predictions=None, steps=None):
        if test_predictions is None:
            predictions = self.test_predictions(steps=steps)
        else:
            predictions = test_predictions
        actual = self.data.test_flow.labels

        #  Make predictions from top test features generated by the base from
        #  the test data.

        # Prepare output format for sklearn reporting
        categories = self.data.categories
        # Prepare actual ground truth
        actual_n = np.argmax(self.data.test_labels, axis=1)
        actual = categories[actual_n]
        # Prepare predicted estimates
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