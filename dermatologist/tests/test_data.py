import unittest
from dermatologist.data import Generator, Data
import numpy as np
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'

Height = 192
Width = 192
Channels = 3

class TestData(unittest.TestCase):

    def setUp(self):
        self.data_dir = os.path.join('dermatologist', 'data')
        self.random_state = 1
        # Get raw data and save processed data set
        Data().save_data()

    def test_init(self):
        """Initialization and data loading."""
        data = Generator()
        #  Test class
        self.assertIsInstance(data, Generator)
        # Tensor shapes
        self.assertEqual(data.train_data.shape[1:], (Height, Width, Channels))
        self.assertEqual(data.test_data.shape[1:], (Height, Width, Channels))
        # Class label list lengths
        self.assertEqual(data.train_data.shape[0], data.train_labels.shape[0])
        self.assertEqual(data.test_data.shape[0], data.test_labels.shape[0])
        # Categories
        self.assertEqual(data.categories.shape[0], data.train_labels.shape[1])
        self.assertEqual(data.categories.shape[0], data.test_labels.shape[1])

class TestRawData(unittest.TestCase):

    def setUp(self):
        self.data_dir = os.path.join('dermatologist', 'data')
        self.random_state = 1

    def test_save(self):
        """Saving of data files."""
        data = Data()

        # Train, test files
        train_path = os.path.join(self.data_dir, 'train.csv')
        valid_path = os.path.join(self.data_dir, 'valid.csv')
        test_path = os.path.join(self.data_dir, 'test.csv')

        # If exists then delete for test
        if os.path.exists(train_path):
            os.remove(train_path)
        # If exists then delete for valid
        if os.path.exists(valid_path):
            os.remove(valid_path)
        # If exists then delete for test
        if os.path.exists(test_path):
            os.remove(test_path)

        #  Create files
        data.save_data_sets()

        # Test for cache file exist
        self.assertTrue(os.path.exists(train_path))
        # Test for cache file exist
        self.assertTrue(os.path.exists(test_path))
        # Test for cache file exist
        self.assertTrue(os.path.exists(category_path))






