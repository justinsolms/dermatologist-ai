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
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import CSVLogger
from keras.applications import VGG16, Xception
from keras.models import Model
from keras.optimizers import Adam

class Model(object):

    output_path = 'output'
    log_path = 'log.{}.csv'
    history_path = 'history.{}.pkl'
    best_model_path = 'weights.best.{}.hdf5'
    data_dir = os.path.join('dermatologist', 'data')

    def __init__(self, data_object,
                 epochs=100,
                 batch_size=48,
                 n_dense=512,
                 dropout=0.2,
                 learn_rate=0.001,
                 n_outputs=7,
                 ):

        self.data = data_object
        self.dropout = dropout
        self.n_dense = n_dense
        self.n_outputs = n_outputs
        self.epochs = epochs
        self.batch_size = batch_size
        self.learn_rate = learn_rate

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
        input_tensor = Input(shape=self.data.train_data.shape[1:])
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
        top_model = Sequential()
        top_model.add(self.base_model)
        top_model.add(GlobalAveragePooling2D(
            input_shape=base_model.output_shape[1:]))
        top_model.add(Dense(n_dense, activation='relu'))
        top_model.add(Dropout(dropout))
        top_model.add(Dense(n_dense, activation='relu'))
        top_model.add(Dropout(dropout))
        top_model.add(Dense(n_outputs, activation='softmax'))

        #  Set up model training
        logger.info('Compiling  model.')
        top_model.compile(
            optimizer=Adam(lr=self.learn_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'categorical_crossentropy'],
            )
        self.top_model = top_model

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

    def set_base_trainable_layers(block_names=None):
        # Source - Dipanjan (DJ) Sarkar

        if block_names is None:
            self.base_model.trainable = False
        else:
            self.base_model.trainable = True

        # FIXME: Not finished
        set_trainable = False
        for layer in self.base_model.layers:
            if layer.name in ['block5_conv1', 'block4_conv1']:
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False

        layers = [(layer, layer.name, layer.trainable) for layer in self.base_model.layers]
        pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])

    def save_top_features(self):
        """Generate top features from base model and save to pickle files."""

        # Data paths
        train_path = os.path.join(self.data_dir, 'train_features.pkl')
        test_path = os.path.join(self.data_dir, 'test_features.pkl')

        # Compute the inputs to the top
        logger.info('Computing top train feature data.')
        train_features = self.base_model.predict(
            self.data.train_data, verbose=1)
        logger.info('Computing top test feature data.')
        test_features = self.base_model.predict(
            self.data.test_data, verbose=1)

        # Save to pickle files
        logger.info('Saving train top feature set.')
        with open(train_path, 'wb') as stream:
            pickle.dump(train_features, stream)
        logger.info('Saving test top feature set.')
        with open(test_path, 'wb') as stream:
            pickle.dump(test_features, stream)

        self.train_features = train_features
        self.test_features = test_features

    def load_top_features(self):
        """Load generated top features of base model from to pickle files."""

        # Data paths
        train_path = os.path.join(self.data_dir, 'train_features.pkl')
        test_path = os.path.join(self.data_dir, 'test_features.pkl')

        # Load from pickle files
        logger.info('Loading train top feature set.')
        with open(train_path, 'rb') as stream:
            self.train_features = pickle.load(stream)
        logger.info('Loading test top feature set.')
        with open(test_path, 'rb') as stream:
            self.test_features = pickle.load(stream)

    def fit(self):
        # Train the model
        logger.info('Training top model from base features.')
        self.history = self.top_model.fit(
            self.train_features, self.data.train_labels,
            validation_split=0.2,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[
                self.checkpointer,
                self.stopper,
                self.csv_logger,
                ],
            verbose=0)

        #  Load best model weights
        path = os.path.join(self.output_path, self.best_model_path)
        if os.path.exists(path):
            self.top_model.load_weights(path)

    def predict(self):
        pass

    def report(self):
        #  Make predictions from top test features generated by the base from
        #  the test data.
        predictions = list()
        for feature in self.test_features:
            predictions.append(
                self.top_model.predict(np.expand_dims(feature, axis=0))
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