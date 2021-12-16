#### Required librairies ####
import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import make_scorer # this function make_scorer will allow to make understable our score for a grid search cv for instance
from sklearn.metrics import f1_score, fbeta_score # These are the score f_score and fbeta_score used to make our score
from rampwf.score_types.classifier_base import ClassifierBaseScoreType
import warnings
warnings.filterwarnings('ignore')

#### title of the problem ####
problem_title = 'Thyroid desease classification'

#### name of the labels ####
_prediction_label_names = [1, 2, 3]

#### task problem ####
Predictions = rw.prediction_types.make_multiclass(
label_names=_prediction_label_names)
workflow = rw.workflows.FeatureExtractorClassifier()


class F_balanced_score(ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='f_balanced_score', precision=3, beta=0.95):
        self.name = name
        self.precision = precision
        self.beta = 0.95

    def __call__(self, y_true, y_pred):
        f_score = f1_score(y_true=y_true, y_pred=y_pred, labels=[0, 1], average=None)
        f_beta_score = fbeta_score(y_true=y_true, y_pred=y_pred,beta=self.beta,labels=[2], average=None)
        res =(f_score[0] + f_score[1] + f_beta_score[0])/3
        return res

score_types = [F_balanced_score(),rw.score_types.Accuracy(name='acc')]

#### train and validation set ####
def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=8, test_size=0.2, random_state=22)
    return cv.split(X, y)

#### get the data ####
_target_column_name = 'disease'

def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name))
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name], axis=1)
    return X_df, y_array


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)