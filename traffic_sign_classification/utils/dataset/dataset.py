# Set up session:
import pickle
import numpy as np
import pandas as pd

class Dataset:
    def __init__(
        self,
        train_pickle,
        valid_pickle,
         test_pickle,
        label_encoding
    ):
        # Load dataset:
        with open(train_pickle, mode='rb') as f:
            self.train = pickle.load(f)
        with open(valid_pickle, mode='rb') as f:
            self.valid = pickle.load(f)
        with open(test_pickle, mode='rb') as f:
            self.test = pickle.load(f)

        # Load label encoding:
        self.label_encoding = pd.read_csv(label_encoding)

        # Dataset dimensions:
        (_, H, W, C) = self.train["features"].shape
        self.IMAGE_SIZE = (H, W, C)
        self.N_CLASSES = len(self.label_encoding)

    def get_label_name(
        self,
        labels
    ):
        """ Get label names:
        """
        return self.label_encoding.ix[labels, 'SignName'].values

    def __str__(self):
        M_TRAIN = self.train["features"].shape[0]
        M_VALID = self.valid["features"].shape[0]
        M_TEST  = self.test["features"].shape[0]

        SIZES = np.array([M_TRAIN, M_VALID, M_TEST])
        feature_size_summary = "Number of training/validation/testing examples = {}".format(
            repr(tuple(SIZES))
        )

        percentange = 100.0 * SIZES / SIZES.sum()
        feature_percentange_summary = "Training/validation/testing percentange sizes = ({:2.2f}, {:2.2f}, {:2.2f})".format(
            percentange[0],
            percentange[1],
            percentange[2]
        )

        feature_sample_summary = "Image data shape = {}".format(
            repr(
                self.IMAGE_SIZE
            )
        )

        label_summary = "Number of classes = {}".format(self.N_CLASSES)

        return "{}\n{}\n{}\n{}\n".format(
            feature_size_summary,
            feature_percentange_summary,
            feature_sample_summary,
            label_summary
        )
