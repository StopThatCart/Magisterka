# %%
import pandas as pd
import numpy as np
from numpy import where, mean, std
import nltk, string, math
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import make_scorer, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_validate, train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.datasets import make_classification
from itertools import product
from imblearn.over_sampling import ADASYN, RandomOverSampler, BorderlineSMOTE, SMOTE, SVMSMOTE, SMOTENC, BorderlineSMOTE

from imblearn.combine import SMOTETomek, SMOTEENN

from collections import Counter
from openpyxl import Workbook

RNG = np.random.RandomState(42)
CLASS = "CLASS"
SAMPLE_ID = "sample_id"

models = {
    "DecisionTree": DecisionTreeClassifier(random_state=RNG),
    "AdaBoost": AdaBoostClassifier(n_estimators=25, random_state=RNG),
    "RandomForest": RandomForestClassifier(random_state=RNG, n_jobs=-1),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RNG, max_iter=10, max_depth=3, learning_rate=0.3)
}
oversamplers = {
    "OverSampler": RandomOverSampler(random_state=RNG),
    "SMOTE": SMOTE(k_neighbors=1, random_state=RNG),
    "BorderlineSmote": BorderlineSMOTE(k_neighbors=1, random_state=RNG),
    "SVMSMOTE": SVMSMOTE(k_neighbors=1, random_state=RNG)
}


estimator = LogisticRegression(solver="liblinear", max_iter=5000, random_state=RNG)

rfe = RFE(estimator=estimator, n_features_to_select=200, step=0.1)

print(rfe.n_features_to_select)