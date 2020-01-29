import unittest
from dermatologist.data import Data
from dermatologist.models import Model

import os

os.environ['KERAS_BACKEND'] = 'tensorflow'

Height = 192
Width = 256
Channels = 3

class TestModel(unittest.TestCase):

    def setUp(self):
        # Prepare data
        self.data_obj = Data()

    def test_init(self):
        """Initialization and model and data preparation."""
        model = Model(self.data_obj, epochs=10, batch_size=10)
        # Test class
        self.assertIsInstance(model, Model)

        data = self.data
        # Tensor shapes
        self.assertEqual(data.train_data.shape[1:], (Height, Width, Channels))
        self.assertEqual(data.test_data.shape[1:], (Height, Width, Channels))
        # Class label list lengths
        self.assertEqual(data.train_data.shape[0], data.train_labels.shape[0])
        self.assertEqual(data.test_data.shape[0], data.test_labels.shape[0])
        # Categories
        self.assertEqual(data.categories.shape[0], data.train_labels.shape[1])
        self.assertEqual(data.categories.shape[0], data.test_labels.shape[1])

    def test_workflow(self):
        """Learning, fitting & reporting"""
        model = Model(self.data_obj, epochs=10, batch_size=10)
        # Test class
        self.assertIsInstance(model, Model)
        # Test fit
        model.fit()
        # Test predictions
        model.predict()
        # Test Reports
        model.report()
