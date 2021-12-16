from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = Pipeline([
            ('classifier', RandomForestClassifier(max_depth=10, min_samples_split=6, 
                             min_samples_leaf=4,random_state=0))
        ])

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)