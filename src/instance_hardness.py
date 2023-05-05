"""
copy from
https://github.com/ghnunes/Curriculum-Learning/blob/53f77fc9662cc7df1740c32939b17418a7a0024d/instance_hardness.py
"""
import numpy as np
import os
import pandas as pd
import pickle
from pyhard import measures
# from src import pyhard_measures
from utils.helper import *
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sys import stdout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
import copy
import warnings
warnings.filterwarnings("ignore")
import sys
sys.dont_write_bytecode = True
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, \
    BaggingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, matthews_corrcoef
from imodels import GreedyRuleListClassifier, BoostedRulesClassifier
from tqdm import tqdm
from src import CFS
from src.classifiers_HPO import *


class customMLPClassifer(MLPClassifier):
    def resample_with_replacement(self, X_train, y_train, sample_weight):

        # normalize sample_weights if not already
        sample_weight = sample_weight / sample_weight.sum(dtype="float")

        X_train_resampled = np.zeros((len(X_train), len(X_train[0])), dtype="float")
        y_train_resampled = np.zeros((len(y_train)), dtype="int")
        for i in range(len(X_train)):
            # draw a number from 0 to len(X_train)-1
            draw = np.random.choice(np.arange(len(X_train)), p=sample_weight)

            # place the X and y at the drawn number into the resampled X and y
            X_train_resampled[i] = X_train[draw]
            y_train_resampled[i] = y_train[draw]

        return X_train_resampled, y_train_resampled

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            X, y = self.resample_with_replacement(X, y, sample_weight)

        return self._fit(X, y, incremental=(self.warm_start and hasattr(self, "classes_")))


# ih_kwargs = {'kDN': {'k': 5, 'distance': 'minkowski'},
#              'DS': {},
#              'DCP': {},
#              'TD_P': {},
#              'TD_U': {},
#              'CL': {},
#              'CLD': {},
#              'MV': {},
#              'CB': {},
#              'N1': {},
#              'N2': {'distance': 'minkowski'},
#              'LSC': {'distance': 'minkowski'},
#              'LSR': {'distance': 'minkowski'},
#              'Harmfulness': {},
#              'Usefulness': {},
#              'F1': {},
#              'F2': {},
#              'F3': {},
#              'F4': {}
#              }


default_models_dic = {
                      'NB': GaussianNB(),                       # Statistical Methods
                      'Ridge': RidgeClassifier(),
                      'LR': LogisticRegression(),
                      'KNN': KNeighborsClassifier(),            # Lazy-Learning Methods
                      'CART': DecisionTreeClassifier(),         # Decision Tree Methods
                      'linSVM': SVC(kernel='linear'),           # Support Vector Machine Methods
                      'rbfSVM': SVC(kernel='rbf'),
                      'MLP1': MLPClassifier(hidden_layer_sizes=(10, ), activation='logistic'),
                      'MLP2': MLPClassifier(hidden_layer_sizes=(10, 10, ), activation='logistic'),
                      'Voting': VotingClassifier(               # Ensemble Learning Methods
                          estimators=[("lr", LogisticRegression()),
                                      ("svm", SVC()),
                                      ("nb", GaussianNB()),
                                      ("mlp", MLPClassifier()),
                                      ("dt", DecisionTreeClassifier())
                                      ]),
                      'RF': RandomForestClassifier(),
                      'ExtraTrees': ExtraTreesClassifier(),
                      'GBDT': GradientBoostingClassifier(),
                      'DTBoosting': AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), algorithm="SAMME"),
                      'LRBoosting': AdaBoostClassifier(base_estimator=LogisticRegression(), algorithm="SAMME"),
                      'SVMBoosting': AdaBoostClassifier(base_estimator=SVC(), algorithm="SAMME"),
                      'MLPBoosting': AdaBoostClassifier(base_estimator=customMLPClassifer(), algorithm="SAMME"),
                      # 'NBBoosting': AdaBoostClassifier(base_estimator=GaussianNB(), algorithm="SAMME"),
                      'DTBagging': BaggingClassifier(base_estimator=DecisionTreeClassifier()),
                      'LRBagging': BaggingClassifier(base_estimator=LogisticRegression()),
                      'SVMBagging': BaggingClassifier(base_estimator=SVC()),
                      'MLPBagging': BaggingClassifier(base_estimator=MLPClassifier()),
                      'NBBagging': BaggingClassifier(base_estimator=GaussianNB()),
                      'Boosted-RS': BoostedRulesClassifier(),   # Rule-Based Methods
                      'Greedy-RL': GreedyRuleListClassifier(),
                      }


def hm_scores(X, y, measures_list):
    """
    Calculates the instance hardness measure of the data
    Parameters
    ----------
    X: feature vectors of data set
    y_train: labels of data set
    measures_list: name of the instance hardness measure being used
    Returns
    -------
    """

    # Converting features in a pandas dataframe
    data_df = pd.concat([X, y], axis=1)

    # Creating measures class
    column_names = ['f_' + str(i + 1) for i in range(len(data_df.columns) - 1)]
    column_names.append('label')

    data_df.columns = column_names

    dc_measures = measures.ClassificationMeasures(data_df, target_col='label')

    # data_scores = []
    # for ih_name in ih_names:
    #     if ih_name == 'TD_P' and X.shape[1] == 37:
    #         print()
    #     dc_measure = dc_measures._call_method(dc_measures._measures_dict.get(ih_name), **ih_kwargs.get(ih_name, {}))
    #     data_scores.append(dc_measure)
    return dc_measures.calculate_all(measures_list)


def repeated_cross_validation(X, y, models_dic=default_models_dic):

    # 5*5 cross-validation
    n_splits = 5
    n_repeats = 5

    rsf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)

    model_keys = sorted(models_dic.keys())
    result_dic = {model_name: [] for model_name in model_keys}

    lst = []
    for _ in range(X.shape[0]):
        lst.append(copy.deepcopy(result_dic))

    test_preds = copy.deepcopy(lst)
    test_truths = y.tolist()

    # conduct the 5*5 cross-validation
    for train_index, test_index in tqdm(rsf.split(X, y)):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]

        # data normalizatiopn
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # feature selection
        selected_cols = CFS.cfs(X_train, y_train)
        X_train = X_train[:, selected_cols]
        X_test = X_test[:, selected_cols]

        # data re-sampling
        sampler = SMOTE(random_state=0)
        X_train, y_train = sampler.fit_resample(X_train, y_train)

        # model training
        for model_name in model_keys:
            clf = models_dic[model_name]

            if model_name == "Greedy-RL" or model_name == "Boosted-RS":
                cols = ['f' + str(i+1) for i in range(X_train.shape[1])]
                X_train = pd.DataFrame(X_train, columns=cols)
                X_test = pd.DataFrame(X_test, columns=cols)
                y_train = pd.DataFrame(y_train).astype('int')
                clf.fit(X_train, y_train, feature_names=cols)
                y_pred = clf.predict(X_test)
                y_pred = np.array(y_pred, dtype=int)

            # elif model_name == "RBFNet":
            #     clf.fit(X_train.T, y_train)
            #     y_pred = clf.predict(X_test.T)
            #
            #     print()

            # elif model_name == 'C45Tree':
            #     cols = ['f' + str(i + 1) for i in range(X_train.shape[1])]
            #     X_train = pd.DataFrame(X_train, columns=cols)
            #     X_test = pd.DataFrame(X_test, columns=cols)
            #     y_train_ = y_train.tolist()
            #     y_train_ = ["defective" if y == 1.0 else "non-defective" for y in y_train_]
            #     y_train_ = pd.DataFrame({"Decision": y_train_})
            #     train_df = pd.concat([X_train, y_train_], axis=1)
            #     model = chef.fit(train_df, {'algorithm': 'C4.5'})
            #
            #     y_pred_ = []
            #     for index, instance in X_test.iterrows():
            #         y_pred_.append(chef.predict(model, instance))
            #
            #     y_pred_ = [1.0 if y == "defective" else 0.0 for y in y_pred_]
            #     y_pred = np.array(y_pred_, dtype='int')

            else:
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

            for j in range(y_pred.shape[0]):
                test_preds[test_index[j]][model_name].append(y_pred[j])

    return test_truths, test_preds


def repeated_cross_validation_Opt_Clfs(X, y, model_names):

    # 5*5 cross-validation
    n_splits = 5
    n_repeats = 5

    rsf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)

    result_dic = {model_name: [] for model_name in model_names}

    lst = []
    for _ in range(X.shape[0]):
        lst.append(copy.deepcopy(result_dic))

    test_preds = copy.deepcopy(lst)
    test_pred_probs = copy.deepcopy(lst)
    test_truths = y.tolist()

    # conduct the 5*5 cross-validation
    for train_index, test_index in tqdm(rsf.split(X, y)):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index].reset_index(drop=True), y[test_index].reset_index(drop=True)

        # data normalization
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # feature selection
        selected_cols = CFS.cfs(X_train, y_train)
        X_train = X_train[:, selected_cols]
        X_test = X_test[:, selected_cols]

        # model training
        for model_name in model_names:

            if model_name == 'KNN':
                clf = KNN_Opt(X_train, y_train, metrics='MCC', opt_algo='TPE')
                y_pred = clf.predict(X_test)
                y_pred_prob = np.max(clf.predict_proba(X_test), axis=1)

            elif model_name == 'NB':
                clf = NB_Opt(X_train, y_train, metrics='MCC', opt_algo='TPE')
                y_pred = clf.predict(X_test)
                y_pred_prob = np.max(clf.predict_proba(X_test), axis=1)

            elif model_name == 'CART':
                clf = CART_Opt(X_train, y_train, metrics='MCC', opt_algo='TPE')
                y_pred = clf.predict(X_test)
                y_pred_prob = np.max(clf.predict_proba(X_test), axis=1)

            elif model_name == 'LR':
                clf = LR_Opt(X_train, y_train, metrics='MCC', opt_algo='TPE')
                y_pred = clf.predict(X_test)
                y_pred_prob = np.max(clf.predict_proba(X_test), axis=1)

            elif model_name == 'SVM':
                clf = SVM_Opt(X_train, y_train, metrics='MCC', opt_algo='TPE')
                y_pred = clf.predict(X_test)
                y_pred_prob = np.max(clf.predict_proba(X_test), axis=1)

            elif model_name == 'MLP':
                clf = MLP_Opt(X_train, y_train, metrics='MCC', opt_algo='TPE')
                y_pred = clf.predict(X_test)
                y_pred_prob = np.max(clf.predict_proba(X_test), axis=1)

            elif model_name == 'Boosted_RS':
                cols = ['f' + str(i+1) for i in selected_cols.tolist()]
                clf = Boosted_RS_Opt(X_train, y_train, cols, metrics='MCC', opt_algo='TPE')

                X_test = pd.DataFrame(X_test, columns=cols)
                y_pred = clf.predict(X_test)
                y_pred_prob = np.max(clf.predict_proba(X_test), axis=1)

                y_pred = np.array(y_pred, dtype=int)
                y_pred_prob = np.array(y_pred_prob, dtype=float)

            elif model_name == 'Greedy_RL':
                cols = ['f' + str(i+1) for i in selected_cols.tolist()]
                clf = Greedy_RL_Opt(X_train, y_train, cols, metrics='MCC', opt_algo='TPE')

                X_test = pd.DataFrame(X_test, columns=cols)
                y_pred = clf.predict(X_test)
                y_pred_prob = np.max(clf.predict_proba(X_test), axis=1)

                y_pred = np.array(y_pred, dtype=int)
                y_pred_prob = np.array(y_pred_prob, dtype=float)

            elif model_name == 'RF':
                clf = RF_Opt(X_train, y_train, metrics='MCC', opt_algo='TPE')
                y_pred = clf.predict(X_test)
                y_pred_prob = np.max(clf.predict_proba(X_test), axis=1)

            elif model_name == 'SVMBoosting':
                clf = SVMBoosting_Opt(X_train, y_train, metrics='MCC', opt_algo='TPE')
                y_pred = clf.predict(X_test)
                y_pred_prob = np.max(clf.predict_proba(X_test), axis=1)

            elif model_name == 'MLPBoosting':
                clf = MLPBoosting_Opt(X_train, y_train, metrics='MCC', opt_algo='TPE')
                y_pred = clf.predict(X_test)
                y_pred_prob = np.max(clf.predict_proba(X_test), axis=1)

            for j in range(y_pred.shape[0]):
                test_preds[test_index[j]][model_name].append(y_pred[j])
                test_pred_probs[test_index[j]][model_name].append(y_pred_prob[j])

    return test_truths, test_preds, test_pred_probs


def ih_scores(test_truths, test_preds):

    n_samples = len(test_truths)

    lst = [0 for _ in range(len(test_preds[0]))]

    correct_predictions = []
    for i in range(n_samples):
        correct_predictions.append(copy.deepcopy(lst))

    for i in range(n_samples):
        y_true = test_truths[i]
        for j in range(len(test_preds[0])):
            for k in range(len(test_preds[0][0])):
                if test_preds[i][j][k] == y_true:
                    correct_predictions[i][j] += 1
            correct_predictions[i][j] /= len(test_preds[0][0])

    ih_score = [0 for _ in range(n_samples)]

    for i in range(n_samples):
        for j in range(len(correct_predictions[i])):
            ih_score[i] += correct_predictions[i][j]

        ih_score[i] /= len(correct_predictions[i])

        ih_score[i] = 1 - ih_score[i]

    return ih_score


def rank_data_according_to_ih(hardness_score, reverse=False, random=False):
    """
    Ranks the data according to instance hardness scores
    ----------
    hardness_scores: instance hardness scores of the data
    y_train: labels of the data
    reverse
        False: if larger scores indicate easier examples
        True: if larger scores indicate harder examples
    random: if to randomize the order of the scores
    Returns
    -------
    res: list of index of the data in order of increasing difficulty
    """

    # train_size, _ = hardness_score.shape
    res = np.asarray(sorted(range(len(hardness_score)), key=lambda k: hardness_score[k], reverse=True))

    if reverse:
        res = np.flip(res, 0)

    if random:
        np.random.shuffle(res)

    return res
