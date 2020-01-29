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

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import pickle
import os

from logzero import logger


class _Object(object):
    data_dir = os.path.join('dermatologist', 'data')
    random_state = 1


class Data(_Object):

    def __init__(self):
        train_path = os.path.join(self.data_dir, 'train.pkl')
        test_path = os.path.join(self.data_dir, 'test.pkl')
        category_path = os.path.join(self.data_dir, 'category.pkl')
        logger.info('Reading train data set.')
        with open(train_path, 'rb') as stream:
            self.train_labels, self.train_data = pickle.load(stream)
        logger.info('Reading test data set.')
        with open(test_path, 'rb') as stream:
            self.test_labels, self.test_data = pickle.load(stream)
        logger.info('Reading data categories.')
        with open(category_path, 'rb') as stream:
            self.categories = pickle.load(stream)


class RawData(_Object):

    pickle_file = 'data_set.pkl'

    def __init__(self, image_size=(192, 192), unittest_size=None):

        self.data_set = self._load_raw_data(
            image_size=image_size,
            unittest_size=unittest_size)

    def save_data(self, test_size=0.2):
        train_path = os.path.join(self.data_dir, 'train.pkl')
        test_path = os.path.join(self.data_dir, 'test.pkl')
        category_path = os.path.join(self.data_dir, 'category.pkl')

        # PIL Images to 4D tensor (n_samples, height, width, channels) with
        # normalized values
        logger.info('Converting PIL images to tensor of uint8.')
        images = np.array(list( map(np.array, self.data_set.image.tolist()) ))
        # logger.info('Deleting PIL images.')
        # del self.data_set['image']
        logger.info('Converting images tensor to float16.')
        images = images.astype(np.float16)
        logger.info('Normalizing image tensor.')
        images /= 255

        # One-hot label encoding of image classification
        logger.info('Creating one-hot classification label tensor.')
        labels_str = self.data_set.classification.values.reshape(-1, 1)
        encoder = OneHotEncoder(sparse=False)
        encoder.fit(labels_str)
        labels = encoder.transform(labels_str)
        categories = encoder.categories_[0]

        # Split data sets, stratify on classes.
        logger.info('Split data into train and test sets.')
        data = images
        train_data, test_data, train_labels, test_labels = train_test_split(
            data, labels, test_size=test_size,
            random_state=self.random_state, shuffle=True,
            stratify=labels)

        # Serialize data set
        logger.info('Writing train data set.')
        with open(train_path, 'wb') as stream:
            pickle.dump((train_labels, train_data), stream)
        logger.info('Writing test data set.')
        with open(test_path, 'wb') as stream:
            pickle.dump((test_labels, test_data), stream)
        logger.info('Writing data categories.')
        with open(category_path, 'wb') as stream:
            pickle.dump(categories, stream)

    def _load_raw_data(self, image_size, unittest_size):
        # Data paths
        images_dir = os.path.join(self.data_dir, 'HAM10000_images')
        meta_data_file = 'HAM10000_metadata.csv'
        meta_data_path = os.path.join(self.data_dir, meta_data_file)

        # load Meta-data
        logger.info('Loading meta-data.')
        data = pd.read_csv(meta_data_path)

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
        data['classification'] = data.dx.map(dict_dx)
        data['class_code'] = pd.Categorical(data['classification']).codes

        # #  Reduce the data set size for sane runtimes during unit testing. Use
        # #  stratification to we get a fair representation of categories.
        if unittest_size is not None:
            data, _ = train_test_split(
                data, train_size=unittest_size,
                random_state=self.random_state, shuffle=True,
                stratify=data.classification)

        # Load images
        data['image_path'] = data.image_id.map(
            lambda file: os.path.join(images_dir, file + '.jpg'))

        # Load images
        image_data = list()
        logger.info('Loading and resizing original images.')
        for path in data.image_path:
            image = Image.open(path)
            image_data.append((
                image.resize(image_size, PIL.Image.ANTIALIAS),
                (image.mode, image.width, image.height)
                ))
        image_data = pd.DataFrame.from_records(image_data)
        image_data.columns = ['image', 'image_meta']
        data = pd.concat([data, image_data], axis='columns', sort=False)

        return data

