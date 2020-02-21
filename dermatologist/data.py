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
from copy import copy

from logzero import logger


class CommonObject(object):
    """A common object to be inherited by all classes."""
    data_dir = os.path.join('dermatologist', 'data')
    image_dir = os.path.join(data_dir, 'ham10000_images')
    generator_dir = os.path.join(data_dir, 'image_generator')
    train_meta_csv = os.path.join(data_dir, 'train.csv')
    valid_meta_csv = os.path.join(data_dir, 'valid.csv')
    test_meta_csv = os.path.join(data_dir, 'test.csv')
    category_meta_csv = os.path.join(data_dir, 'categories.csv')
    random_state = 1


class Data(CommonObject):

    def __init__(self,
                 raw_data_obj,
                 test_split=0.2,
                 batch_size=32,
                 target_size=(192, 192),
                 save_to_dir=False,
                 ):
        """Initialization."""
        self._raw_data_obj = raw_data_obj

        # To save generated images or not
        if not save_to_dir:
            self.generator_dir = None

        self.test_split = test_split
        self.batch_size = batch_size
        self.target_size = target_size

        self.shape = target_size + (3,)

        # Load category metadata CSV file
        logger.info('Loading category metadata.')
        category_meta_df = self._raw_data_obj.category_meta_data
        self.n_categories = len(category_meta_df.index)
        self.categories = category_meta_df
        # Use this to make sure all generator's class_indice property use
        # identical class rankings.
        classes = category_meta_df['category'].tolist()

        # Load training metadata CSV file
        logger.info('Loading training metadata.')
        train_meta_df = self._raw_data_obj.train_meta_data
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
            classes=None,
            weight_col=None,
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
        logger.info('Loading validation metadata.')
        valid_meta_df = self._raw_data_obj.valid_meta_data
        # Validation image augmentation generator
        logger.info('Creating validation data generator.')
        self.valid_generator = ImageDataGenerator(
            rescale=1./255,
        )
        self.valid_flow = self.valid_generator.flow_from_dataframe(
            valid_meta_df,
            directory=self.image_dir,
            x_col='file_name',
            y_col='category',
            classes=None,
            weight_col=None,
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
        logger.info('Loading testing metadata.')
        test_meta_df = self._raw_data_obj.test_meta_data
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
            classes=None,
            weight_col=None,
            target_size=self.target_size,
            color_mode='rgb',
            class_mode='categorical',
            batch_size=self.batch_size,
            save_to_dir=self.generator_dir,
            save_prefix='test',
            shuffle=False,  # Do not shuffle for tests
            seed=self.random_state,
            )

    @property
    def class_indice_dict(self):
        """dict: Mapping class name to the class indice (integer)."""
        return self._raw_data_obj.class_indice_dict

    @property
    def class_indice_weight_dict(self):
        """dict: Mapping class indices (integers) to a weight (float) value."""
        return self._raw_data_obj.class_indice_weight_dict


class RawData(CommonObject):

    # Information as provided: HAM10000 dataset - Tschandl, Rosendahl, Kittler
    _dx_class_dict = {
        'mel': 'Melanoma', # malignant
        'akiec': 'Actinic Keratoses', # precursors
        'bcc': 'Basal cell carcinoma', # grows destructively
        'bkl': 'Benign keratosis', # benign
        'nv': 'Melanocytic nevi', # benign
        'df': 'Dermatofibroma', # benign
        'vasc': 'Vascular skin lesions', # benign
        }

    # Deemed importance of classes.
    _dx_weight_dict = {
        'mel': 1.0, # malignant
        'akiec': 1.0, # precursors
        'bcc': 1.0, # grows destructively
        'bkl': 1.0, # benign
        'nv': 1.0, # benign
        'df': 1.0, # benign
        'vasc': 1.0, # benign
        }

    def __init__(self, test_size=0.20):
        self.test_size = test_size

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
        logger.info('Adding further classification meta-data.')
        data['category'] = data.dx.map(self._dx_class_dict)
        data['category_code'] = data.dx.map(self._dx_indice_dict)

        self.data = data

    def _dx_class_tuples(self):
        """Return alpha sorted (dx, class) tuples"""
        dx_class = [(key, value) for key, value in self._dx_class_dict.items()]
        dx_class.sort(key=lambda x: x[1])
        return dx_class

    @property
    def _dx_indice_dict(self):
        """dict: Mapping the dx column to the class indice (integer)."""
        dx, _  = zip(*self._dx_class_tuples())
        class_numbers = [i for i in range(len(dx))]
        return dict(zip(dx, class_numbers))

    @property
    def class_indice_dict(self):
        """dict: Mapping class name to the class indice (integer)."""
        _, class_names  = zip(*self._dx_class_tuples())
        class_numbers = [i for i in range(len(class_names))]
        return dict(zip(class_names, class_numbers))

    @property
    def class_indice_weight_dict(self):
        """dict: Mapping class indices (integers) to a weight (float) value."""
        mapping = {
            self._dx_indice_dict[key]: self._dx_weight_dict[key]
            for key in self._dx_class_dict.keys()
        }

        return mapping

    def create_data_sets(self):
        logger.info('Split data into train and test sets.')
        self.train_meta_data, self.test_meta_data = train_test_split(
            self.data,
            test_size=self.test_size,
            stratify=self.data['category_code'],
            random_state=self.random_state, shuffle=True,
            )
        logger.info('Split train data into train and validation sets.')
        self.train_meta_data, self.valid_meta_data = train_test_split(
            self.train_meta_data,
            test_size=self.test_size,
            stratify=self.train_meta_data['category_code'],
            random_state=self.random_state, shuffle=True,
            )

        logger.info('Write categorical meta data csv file.')
        self.category_meta_data = pd.DataFrame(
            [d for d in self.class_indice_dict.items()],
            columns=['category', 'category_code']
            )
        self.category_meta_data.to_csv(self.category_meta_csv, index=False)

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



