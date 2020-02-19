"""
Module Docstring
"""

__author__ = "Justin Solms"
__version__ = "0.0.0"
__license__ = "GPL"

import numpy as np
import pandas as pd
import PIL
from PIL import Image

import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator

import pickle
import os

from logzero import logger


class CommonObject(object):
    """A common object to be inherited by all classes."""
    data_dir = os.path.join('dermatologist', 'data')
    image_dir = os.path.join(data_dir, 'ham10000_images')
    generator_dir = os.path.join(data_dir, 'image_generator')
    train_meta_csv = os.path.join(data_dir, 'train.csv')
    valid_meta_csv = os.path.join(data_dir, 'valid.csv')
    test_meta_csv = os.path.join(data_dir, 'test.csv')
    random_state = 1


class Data(CommonObject):

    def __init__(self,
                 test_split=0.2,
                 batch_size=32,
                 target_size=(192, 192),
                 save_to_dir=False,
                 ):
        """Initialization."""
        # To save generated images or not
        if not save_to_dir:
            self.generator_dir = None

        self.test_split = test_split
        self.batch_size = batch_size
        self.target_size = target_size

        self.shape = target_size + (3,)

        self.n_classes = 7  # TODO: Get this form the flow

        # Load training metadata CSV file
        logger.info('Loading training metadata from {}'.format(
            self.train_meta_csv))
        train_meta_df = pd.read_csv(self.train_meta_csv)
        # Training image augmentation generator
        logger.info('Creating training data generator.')
        self.train_generator = ImageDataGenerator(
            rescale=1./255,
            # featurewise_center=True,
            # featurewise_std_normalization=True,
            rotation_range=5,
            width_shift_range=0.05,
            height_shift_range=0.05,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.05,
            fill_mode='constant',
            )
        self.train_flow = self.train_generator.flow_from_dataframe(
            train_meta_df,
            directory=self.image_dir,
            x_col='file_name',
            y_col='category',
            weight_col=None,  # TODO: Include weights
            target_size=self.target_size,
            color_mode='rgb',
            class_mode='categorical',
            batch_size=self.batch_size,
            save_to_dir=self.generator_dir,
            save_prefix='train',
            shuffle=True,
            seed=self.random_state,
            )

        # Load validation metadata CSV file
        logger.info('Loading validation metadata from {}'.format(
            self.valid_meta_csv))
        valid_meta_df = pd.read_csv(self.valid_meta_csv)
        # Validation image augmentation generator
        logger.info('Creating validation data generator.')
        self.valid_generator = ImageDataGenerator(
            rescale=1./255,
        )
        self.validation_flow = self.valid_generator.flow_from_dataframe(
            valid_meta_df,
            directory=self.image_dir,
            x_col='file_name',
            y_col='category',
            weight_col=None,  # TODO: Include weights
            target_size=self.target_size,
            color_mode='rgb',
            class_mode='categorical',
            batch_size=self.batch_size,
            save_to_dir=self.generator_dir,
            save_prefix='valid',
            shuffle=True,
            seed=self.random_state,
            )

        # Load testing metadata CSV file
        logger.info('Loading testing metadata from {}'.format(
            self.test_meta_csv))
        test_meta_df = pd.read_csv(self.test_meta_csv)
        # Testing image augmentation generator
        logger.info('Creating testing data generator.')
        self.test_generator = ImageDataGenerator(
            rescale=1./255
            )
        self.test_flow = self.test_generator.flow_from_dataframe(
            test_meta_df,
            directory=self.image_dir,
            x_col='file_name',
            y_col='category',
            weight_col=None,  # FIXME:
            target_size=self.target_size,
            color_mode='rgb',
            class_mode='categorical',
            batch_size=self.batch_size,
            save_to_dir=self.generator_dir,
            save_prefix='test',
            shuffle=False,  # Do not shuffle for tests
            seed=self.random_state,
            )


class RawData(CommonObject):

    def __init__(self, test_size=0.20):

        # Data paths
        meta_data_csv = 'HAM10000_metadata.csv'
        images_dir = os.path.join(self.data_dir, 'HAM10000_images')
        meta_data_path = os.path.join(self.data_dir, meta_data_csv)

        # load Meta-data
        logger.info('Loading meta-data.')
        data = pd.read_csv(meta_data_path)

        # Add image filename column
        logger.info('Add image file-name column.')
        data['file_name'] = data.image_id.map(lambda f: '{}.jpg'.format(f))

        #  Classification
        # Information as provided: HAM10000 dataset - Tschandl, Rosendahl, Kittler
        dict_dx = {
            'mel': 'Melanoma', # malignant
            'akiec': 'Actinic Keratoses', # precursors
            'bcc': 'Basal cell carcinoma', # grows destructively
            'bkl': 'Benign keratosis', # benign
            'nv': 'Melanocytic nevi', # benign
            'df': 'Dermatofibroma', # benign
            'vasc': 'Vascular skin lesions', # benign
            }
        logger.info('Adding further classification meta-data.')
        data['category'] = data.dx.map(dict_dx)
        data['category_code'] = pd.Categorical(data['category']).codes

        logger.info('Split data into train and test sets.')
        self.train_meta_data, self.test_meta_data = train_test_split(
            data,
            test_size=test_size, stratify=data['category_code'],
            random_state=self.random_state, shuffle=True,
            )
        logger.info('Split train data into train and validation sets.')
        self.train_meta_data, self.valid_meta_data = train_test_split(
            self.train_meta_data,
            test_size=test_size, stratify=self.train_meta_data['category_code'],
            random_state=self.random_state, shuffle=True,
            )

        logger.info('Write train and test set meta data csv files.')
        self.train_meta_data.to_csv(self.train_meta_csv)
        self.valid_meta_data.to_csv(self.valid_meta_csv)
        self.test_meta_data.to_csv(self.test_meta_csv)

    # TODO: Include all analysis/exploration plots

    def plot_categories(self):
        # Histograms
        data = self.meta_data
        data.dx_type.value_counts().plot(kind='bar')
        plt.show()
        data.sex.value_counts().plot(kind='bar')
        plt.show()
        data.localization.value_counts().plot(kind='bar')
        plt.show()
        data.classification.value_counts().plot(kind='bar')
        plt.show()



