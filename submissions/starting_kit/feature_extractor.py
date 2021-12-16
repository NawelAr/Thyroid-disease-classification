from sklearn.compose import make_column_transformer
from sklearn import preprocessing
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
import numpy as np

class FeatureExtractor():
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def transform(self, X):
        cols = ["age", "TSH", "T3", "TT4", "T4U", "FTI"]
        ct = make_column_transformer(('passthrough', cols))
        XX = ct.fit_transform(X)
        imp = SimpleImputer(strategy='most_frequent')
        XX_ = imp.fit_transform(XX)
        return XX_