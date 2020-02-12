#!/usr/bin/env python3
"""
Main Docstring
"""

__author__ = "Justin Solms"
__version__ = "0.0.0"
__license__ = "GPL"

from dermatologist.data import Data, RawData
from dermatologist.models import Model

import os
import argparse
from logzero import logger

os.environ['KERAS_BACKEND'] = 'tensorflow'

def main(args):
    """ Main entry point of the app."""
    logger.info(args)

    #  Load and process raw data
    if args.new == True:
        logger.info('Forced to re-create test-train set image metadata.')
        raw_data = RawData()


    # Load model
    logger.info('Initializing Model class.')
    model = Model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_dense=args.n_dense,
        dropout=args.dropout,
        learn_rate=args.learn_rate,
    )

    # Train model
    logger.info('Training top model.')
    model.fit()

    # Make reports
    model.report()


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--new", action='store_true',
        help="Generate and save top features from base",)

    parser.add_argument(
        "--generate", action='store_true',
        help="Generate and save top features from base",)

    parser.add_argument(
        "-e", "--epochs", dest="epochs",
        help="Number of training epochs",
        type=int, default=150,)

    parser.add_argument(
        "-b", "--batch_size", dest="batch_size",
        help="Training batch size",
        type=int, default=16)

    parser.add_argument(
        "-n", "--n_dense", dest="n_dense",
        help="Size of fully connected, trained, dense layer",
        type=int, default=512)

    parser.add_argument(
        "-d", "--dropout", dest="dropout",
        help="Dropout rate of the trained layer",
        type=float, default=0.2)

    parser.add_argument(
        "-l", "--", dest="learn_rate",
        help="Learning rate",
        type=float, default=0.001)

    parser.add_argument(
        "-v", "--verbose", dest="verbose",
        help="Verbosity of training output",
        type=int, default=0)

    args = parser.parse_args()
    main(args)