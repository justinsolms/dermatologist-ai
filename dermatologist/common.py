""" Object components common to all classes.
"""

import os

class CommonObject(object):
    """A common object to be inherited by all classes."""
    data_dir = os.path.join('dermatologist', 'data')
    meta_data_path = os.path.join(data_dir, 'HAM10000_metadata.csv')
    generator_dir = os.path.join(data_dir, 'image_generator')
    train_meta_csv = os.path.join(data_dir, 'train.csv')
    valid_meta_csv = os.path.join(data_dir, 'valid.csv')
    test_meta_csv = os.path.join(data_dir, 'test.csv')
    category_meta_csv = os.path.join(data_dir, 'categories.csv')

    # TODO: Random state as arg switch passed to ALL methods.
    random_state = 1

    def __init__(self):

        # Test to see if if running on FloydHub
        if os.getcwd() == '/floyd/home':
            self.host_is = 'floyd_hub'
            self.input_path = os.path.join('/floyd/input')
            self.output_path = os.path.join('.', 'output' )
        else:
            self.host_is = 'local'
            self.input_path = os.path.join('.', 'input')
            self.output_path = os.path.join('.', 'output' )

        # Depends on where the input is
        self.image_dir = os.path.join(self.input_path, 'ham10000_images')

        #  Make output directory if required
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
