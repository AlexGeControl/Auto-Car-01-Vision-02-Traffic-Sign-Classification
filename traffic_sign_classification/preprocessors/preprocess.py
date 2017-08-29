# Set up session:
import numpy as np
import cv2

# Sklearn transformer interface:
from sklearn.base import TransformerMixin

class Preprocessor(TransformerMixin):
    """ Traffic sign classification image preprocessing
    """
    def __init__(
        self,
        grayscale = False
    ):
        self.grayscale = grayscale

    def transform(self, X):
        """
        """
        return np.array(
            [self._transform(x) for x in X]
        )

    def fit(self, X, y=None):
        return self

    def set_params(self, **kwargs):
        self.__dict__.update(kwargs)

    def _transform(self, X):
        """ Convert image to YUV color space, then equalize its histogram:
        """
        # Parse image dimensions:
        H, W, C = X.shape

        # Convert to YUV:
        YUV = cv2.cvtColor(X, cv2.COLOR_RGB2YUV)

        # Extract Y channel and equalize its histogram:
        Y = cv2.split(YUV)[0]

        return cv2.equalizeHist(Y).reshape((H, W, 1))
