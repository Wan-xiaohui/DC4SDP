from utils.helper import *
import pickle
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import matthews_corrcoef
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
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from imodels import GreedyRuleListClassifier, BoostedRulesClassifier
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
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
from self_paced_ensemble import SelfPacedEnsembleClassifier
from self_paced_ensemble.canonical_ensemble import *
from self_paced_ensemble.canonical_resampling import *
from tqdm import tqdm
from src import CFS
from src.classifiers_HPO import *
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.preprocessing import binarize
from imblearn.over_sampling import SMOTE as SMOTE_IMB
from scipy.stats import wilcoxon
from utils.cliffsDelta import cliffsDelta
from utils.calculate_cliff import calculate_cliff


"""
An Ensemble Generation Method Based on Instance Hardness
https://arxiv.org/pdf/1804.07419.pdf
"""


def hm_scores(X, y, measure_name):
    """
    Calculates the instance hardness measure of the data
    Parameters
    ----------
    X: feature vectors of data set
    y_train: labels of data set
    ih_name: name of the instance hardness measure being used
    Returns
    -------
    """

    # Converting features in a pandas dataframe
    data_df = pd.concat([pd.DataFrame(X), pd.Series(y)], axis=1)

    # Creating measures class
    column_names = ['f_' + str(i + 1) for i in range(len(data_df.columns) - 1)]
    column_names.append('label')

    data_df.columns = column_names
    dc_measures = measures.ClassificationMeasures(data_df, target_col='label')

    return dc_measures.calculate_all(measures_list=[measure_name])


def calc_performance(label_true, label_pred):
    MCC = matthews_corrcoef(label_true, label_pred)
    return MCC


class HMSMOTEBaggingClassifier():
    def __init__(self,
                 n_samples=100,
                 min_ratio=1.0,
                 with_replacement=True,
                 base_estimator=None,
                 n_estimators=10,
                 hm_name='kDN',
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 random_state=None):
        self.n_estimators = n_estimators
        self.model_list = []
        self.hm_name = hm_name

    def fit(self, X, y, minority_target=None):
        if minority_target is None:
            # Determine the minority class label.
            stats_c_ = Counter(y)
            maj_c_ = max(stats_c_, key=stats_c_.get)
            min_c_ = min(stats_c_, key=stats_c_.get)
            self.minority_target = min_c_
        else:
            self.minority_target = minority_target

        minus_dm = 1 - hm_scores(X, y, self.hm_name)

        self.model_list = []
        df = pd.DataFrame(X)
        df['label'] = y
        df_maj = df[df['label'] != self.minority_target]
        maj_weights = 1 / df_maj.shape[0] + minus_dm[df['label'] != self.minority_target]
        maj_probs = maj_weights.values / np.sum(maj_weights.values)
        maj_probs = np.squeeze(maj_probs).tolist()
        df_min = df[df['label'] == self.minority_target]
        min_weights = 1 / df_min.shape[0] + minus_dm[df['label'] == self.minority_target]
        min_probs = min_weights.values / np.sum(min_weights.values)
        min_probs = np.squeeze(min_probs).tolist()

        cols = df.columns.tolist()
        cols.remove('label')

        for ibagging in range(self.n_estimators):
            # b = min(0.1 * ((ibagging % 10) + 1), 1)
            train_maj = df_maj.sample(frac=1.0, weights=maj_probs, replace=True)
            train_min = df_min.sample(frac=1.0, weights=min_probs, replace=True)

            df_k = train_maj.append(train_min)
            X_train, y_train = SMOTE_IMB(k_neighbors=min(3, len(train_min) - 1)).fit_resample(df_k[cols], df_k['label'])
            model = DT().fit(X_train, y_train)
            self.model_list.append(model)
        return self

    def predict_proba(self, X):
        y_pred = np.array([model.predict(X) for model in self.model_list]).mean(axis=0)
        if y_pred.ndim == 1:
            y_pred = y_pred[:, np.newaxis]
        if y_pred.shape[1] == 1:
            y_pred = np.append(1 - y_pred, y_pred, axis=1)
        return y_pred

    def predict(self, X):
        y_pred_binarazed = binarize(self.predict_proba(X)[:, 1].reshape(1, -1), threshold=0.5)[0]
        return y_pred_binarazed


def repeated_cross_validation_HMBagging(X, y, model_names):

    n_splits = 5
    n_repeats = 5

    rsf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)

    result_dic = {model_name: [] for model_name in model_names}

    for train_index, test_index in tqdm(rsf.split(X, y)):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index].reset_index(drop=True), y[test_index].reset_index(drop=True)

        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        selected_cols = CFS.cfs(X_train, y_train)
        X_train = X_train[:, selected_cols]
        X_test = X_test[:, selected_cols]

        for model_name in model_names:

            if model_name == 'SMOTEBagging':
                model = SMOTEBaggingClassifier(
                    base_estimator=DecisionTreeClassifier(),
                    n_estimators=50,
                    random_state=0
                )

            if model_name == 'HMSMOTEBagging_kDN':
                model = HMSMOTEBaggingClassifier(
                    base_estimator=DecisionTreeClassifier(),
                    hm_name='kDN',
                    n_estimators=50,
                    random_state=0
                )

            if model_name == 'HMSMOTEBagging_DS':
                model = HMSMOTEBaggingClassifier(
                    base_estimator=DecisionTreeClassifier(),
                    hm_name='DS',
                    n_estimators=50,
                    random_state=0
                )

            if model_name == 'HMSMOTEBagging_DCP':
                model = HMSMOTEBaggingClassifier(
                    base_estimator=DecisionTreeClassifier(),
                    hm_name='DCP',
                    n_estimators=50,
                    random_state=0
                )

            if model_name == 'HMSMOTEBagging_CL':
                model = HMSMOTEBaggingClassifier(
                    base_estimator=DecisionTreeClassifier(),
                    hm_name='CL',
                    n_estimators=50,
                    random_state=0
                )

            if model_name == 'HMSMOTEBagging_LSC':
                model = HMSMOTEBaggingClassifier(
                    base_estimator=DecisionTreeClassifier(),
                    hm_name='LSC',
                    n_estimators=50,
                    random_state=0
                )

            if model_name == 'HMSMOTEBagging_U':
                model = HMSMOTEBaggingClassifier(
                    base_estimator=DecisionTreeClassifier(),
                    hm_name='U',
                    n_estimators=50,
                    random_state=0
                )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mcc = calc_performance(y_test, y_pred)
            result_dic[model_name].append(mcc)

    return result_dic


def cv_prediction_using_hm(path_to_datasets, path_to_saved_file, model_names):

    data_list, label_list, fname = load_data(path_to_datasets)

    for index, file in enumerate(fname):

        print('File:\t' + file + '...', flush=True)

        try:
            data = data_list[index]
            label = label_list[index]

            dataset = pd.DataFrame(np.column_stack((data, label)))

            X = dataset.iloc[:, :-1]
            y = dataset.iloc[:, -1]

            result_dic = repeated_cross_validation_HMBagging(X, y, model_names)

            pkfile = open(path_to_saved_file + file + '.pickle', 'wb')
            pickle.dump(result_dic, pkfile)

        except Exception as e:
            print('The following error occurred in case of the dataset {}: \n{}'.format(file, e))


def get_prediction_results(path_to_datasets, path_to_saved_file, path_to_saved_csv):
    _, _, fname = load_data(path_to_datasets)

    for index, file in enumerate(fname):
        try:
            pkfile = open(path_to_saved_file + file + '.pickle', 'rb')
            result_dic = pickle.load(pkfile)
            result_df = pd.DataFrame(result_dic)
            result_df.to_csv(path_to_saved_csv + file + '.csv')

        except:
            print(file + '.pickle' + " does not exist !")


def calc_avg_performance(path_to_datasets, path_to_saved_csv):
    _, _, fname = load_data(path_to_datasets)

    for index, file in enumerate(fname):
        try:
            result_df = pd.read_csv(path_to_saved_csv + file + '.csv')
            cols_to_drop = [col for col in result_df.columns if 'Unnamed' in col]  # 找到所有“Unnamed”列
            result_df = result_df.drop(cols_to_drop, axis=1)  # 删除“Unnamed”列

            if index == 0:
                avg_mcc = result_df.mean()
            else:
                avg_mcc = pd.concat([avg_mcc, result_df.mean()], axis=1)

        except:
            print(file + '.pickle' + " does not exist !")

    avg_mcc.T.to_csv(path_to_saved_csv + "total_avg_mcc.csv")


if __name__ == '__main__':

    model_names = ['SMOTEBagging', 'HMSMOTEBagging_kDN', 'HMSMOTEBagging_DS', 'HMSMOTEBagging_DCP',
                   'HMSMOTEBagging_CL', 'HMSMOTEBagging_LSC', 'HMSMOTEBagging_U']

    path_to_datasets = 'datasets/'
    path_to_saved_file = 'dump/cv_results_hm/'
    path_to_saved_csv = 'dump/prediction_results_hm/'

    cv_prediction_using_hm(path_to_datasets, path_to_saved_file, model_names)
    get_prediction_results(path_to_datasets, path_to_saved_file, path_to_saved_csv)
    calc_avg_performance(path_to_datasets, path_to_saved_csv)