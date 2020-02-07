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

import pickle
import os

from logzero import logger


class _Object(object):
    data_dir = os.path.join('dermatologist', 'data')
    train_meta_csv = os.path.join(data_dir, 'train.csv')
    test_meta_csv = os.path.join(data_dir, 'test.csv')
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

    def __init__(self, test_size=0.20):

        # Data paths
        meta_data_csv = 'HAM10000_metadata.csv'
        images_dir = os.path.join(self.data_dir, 'HAM10000_images')
        meta_data_path = os.path.join(self.data_dir, meta_data_csv)

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
        data['category'] = data.dx.map(dict_dx)
        data['category_code'] = pd.Categorical(data['category']).codes

        logger.info('Split data into train and test sets.')
        self.train_meta_data, self.test_meta_data = train_test_split(
            data, test_size=test_size, stratify=data['category_code'],
            random_state=self.random_state, shuffle=True,
            )

        logger.info('Write train and test set meta data csv files.')
        self.train_meta_data.to_csv(self.train_meta_csv)
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



