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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.utils import shuffle
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
import numpy as np
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from src import CFS
from src.ReliefF import ReliefF
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from self_paced_ensemble.canonical_resampling.mahakil import MAHAKIL
import problexity as px
from problexity.classification.neighborhood import n2
from problexity.classification.feature_based import f1
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances
from utils.helper import *


class Error(Exception):
    pass

dc_measures = ["f1", "f1v", "f2", "f3", "f4", "l1", "l2", "l3", "n1", "n2", "n3", "n4", "t1",
               "lsc", "density", "clsCoef", "hubs", "t2", "t3", "t4", "c1", "c2"]


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


def opt_norm(X_train, y_train, X_test):
    n2_min = 1.0
    for normalization_type in ["standard", "none"]:

        if normalization_type == "standard":
            # 数据归一化处理
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train_ = scaler.transform(X_train)

            n2_value = n2(X_train_, y_train)
            # cc = px.ComplexityCalculator()
            # cc.fit(X_train_, y_train)
            # n2_value = cc.report()['complexities']['n2']

            if n2_value < n2_min:
                n2_min = n2_value
                opt_X_train = X_train_
                opt_X_test = scaler.transform(X_test)
                optimal_norm_type = "standard"

        elif normalization_type == "min-max":
            # 数据归一化处理
            scaler = preprocessing.MinMaxScaler().fit(X_train)
            X_train_ = scaler.transform(X_train)
            n2_value = n2(X_train_, y_train)

            if n2_value < n2_min:
                n2_min = n2_value
                opt_X_train = X_train_
                opt_X_test = scaler.transform(X_test)
                optimal_norm_type = "min-max"

        elif normalization_type == "none":
            n2_value = n2(X_train, y_train)

            if n2_value < n2_min:
                n2_min = n2_value
                opt_X_train = X_train
                opt_X_test = X_test
                optimal_norm_type = "none"
    # print(optimal_norm_type)
    return opt_X_train, opt_X_test


def opt_fs(X_train, y_train, X_test):

    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.to_numpy()

    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.to_numpy()

    f1_min = 1.0
    for fs_type in ["SKB_anova", "SKB_mutual", "Tree", "None"]:

        # feature selection
        if fs_type == "CFS":
            selected_cols = CFS.cfs(X_train, y_train)
            X_train_ = X_train[:, selected_cols]

            f1_value = f1(X_train_, y_train)
            if f1_value < f1_min:
                f1_min = f1_value
                opt_X_train = X_train_
                opt_X_test = X_test[:, selected_cols]

        elif fs_type == "SKB_anova":
            n_features = int(0.7*X_train.shape[1])
            selector = SelectKBest(f_classif, k=n_features)
            selector.fit(X_train, y_train)
            selected_cols = selector.get_support(indices=True)
            X_train_ = X_train[:, selected_cols]

            f1_value = f1(X_train_, y_train)
            if f1_value < f1_min:
                f1_min = f1_value
                opt_X_train = X_train_
                opt_X_test = X_test[:, selected_cols]

        elif fs_type == "SKB_mutual":
            n_features = int(0.7*X_train.shape[1])
            selector = SelectKBest(mutual_info_classif, k=n_features)
            selector.fit(X_train, y_train)
            selected_cols = selector.get_support(indices=True)
            X_train_ = X_train[:, selected_cols]

            f1_value = f1(X_train_, y_train)
            if f1_value < f1_min:
                f1_min = f1_value
                opt_X_train = X_train_
                opt_X_test = X_test[:, selected_cols]

        elif fs_type == "ReliefF":
            n_features = int(0.7*X_train.shape[1])
            selector = ReliefF(n_features_to_keep=n_features)
            selector.fit(X_train, y_train)
            selected_cols = selector.top_features[0:selector.n_features_to_keep]
            X_train_ = X_train[:, selected_cols]

            f1_value = f1(X_train_, y_train)
            if f1_value < f1_min:
                f1_min = f1_value
                opt_X_train = X_train_
                opt_X_test = X_test[:, selected_cols]

        elif fs_type == "LinSVM":
            clf = LinearSVC()
            clf.fit(X_train, y_train)
            model = SelectFromModel(clf, prefit=True)
            selected_cols = model.get_support(indices=True)
            X_train_ = X_train[:, selected_cols]

            f1_value = f1(X_train_, y_train)
            if f1_value < f1_min:
                f1_min = f1_value
                opt_X_train = X_train_
                opt_X_test = X_test[:, selected_cols]

        elif fs_type == "Tree":
            clf = ExtraTreesClassifier()
            clf.fit(X_train, y_train)
            model = SelectFromModel(clf, prefit=True)
            selected_cols = model.get_support(indices=True)
            X_train_ = X_train[:, selected_cols]

            f1_value = f1(X_train_, y_train)
            if f1_value < f1_min:
                f1_min = f1_value
                opt_X_train = X_train_
                opt_X_test = X_test[:, selected_cols]

        elif fs_type == "None":
            f1_value = f1(X_train, y_train)
            if f1_value < f1_min:
                f1_min = f1_value
                opt_X_train = X_train
                opt_X_test = X_test

    return opt_X_train, opt_X_test


def opt_rs(X_train, y_train):
    f1_min = 1.0
    for rs_type in ["SMOTE", "SMOTETomek", "SMOTEENN"]:

        # data re-sampling
        if rs_type == "SMOTE":
            sampler = SMOTE(random_state=42)
            X_train_, y_train_ = sampler.fit_resample(X_train, y_train)

            f1_value = f1(X_train_, y_train_)
            if f1_value < f1_min:
                f1_min = f1_value
                opt_X_train = X_train_
                opt_y_train = y_train_

        elif rs_type == "BorderSMOTE":
            sampler = BorderlineSMOTE(random_state=42)
            X_train_, y_train_ = sampler.fit_resample(X_train, y_train)

            f1_value = f1(X_train_, y_train_)
            if f1_value < f1_min:
                f1_min = f1_value
                opt_X_train = X_train_
                opt_y_train = y_train_

        elif rs_type == "SMOTETomek":
            sampler = SMOTETomek(random_state=42)
            X_train_, y_train_ = sampler.fit_resample(X_train, y_train)

            f1_value = f1(X_train_, y_train_)
            if f1_value < f1_min:
                f1_min = f1_value
                opt_X_train = X_train_
                opt_y_train = y_train_

        elif rs_type == "SMOTEENN":
            sampler = SMOTEENN(random_state=42)
            X_train_, y_train_ = sampler.fit_resample(X_train, y_train)

            f1_value = f1(X_train_, y_train_)
            if f1_value < f1_min:
                f1_min = f1_value
                opt_X_train = X_train_
                opt_y_train = y_train_

        else:
            raise Error('Unexpected \'by\' type {}'.format(rs_type))

    return opt_X_train, opt_y_train


def repeated_cross_validation_DCMPreProcessing(X, y, preprocessing_types):

    n_splits = 5
    n_repeats = 5

    rsf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)

    result_dic = {preprocessing_type: [] for preprocessing_type in preprocessing_types}

    for preprocessing_type in preprocessing_types:

        for train_index, test_index in tqdm(rsf.split(X, y)):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y[train_index].reset_index(drop=True), y[test_index].reset_index(drop=True)

            if preprocessing_type == "default":
                scaler = StandardScaler().fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)

                selected_cols = CFS.cfs(X_train, y_train)
                X_train = X_train[:, selected_cols]
                X_test = X_test[:, selected_cols]

                smo = SMOTE(random_state=42)
                X_train, y_train = smo.fit_resample(X_train, y_train)

            elif preprocessing_type == "adaptive":
                # select the optimal data normalization type
                X_train, X_test = opt_norm(X_train, y_train, X_test)

                # # select the optimal feature selection type
                X_train, X_test = opt_fs(X_train, y_train, X_test)

                # select the optimal data resampling type
                X_train, y_train = opt_rs(X_train, y_train)

            # model training
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mcc = calc_performance(y_test, y_pred)
            result_dic[preprocessing_type].append(mcc)

    return result_dic


def cv_prediction_using_dcm(path_to_datasets, path_to_saved_file, preprocessing_types):

    data_list, label_list, fname = load_data(path_to_datasets)

    for index, file in enumerate(fname):

        print('File:\t' + file + '...', flush=True)

        try:
            data = data_list[index]
            label = label_list[index]

            dataset = pd.DataFrame(np.column_stack((data, label)))

            X = dataset.iloc[:, :-1]
            y = dataset.iloc[:, -1]

            result_dic = repeated_cross_validation_DCMPreProcessing(X, y, preprocessing_types)

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
            cols_to_drop = [col for col in result_df.columns if 'Unnamed' in col]
            result_df = result_df.drop(cols_to_drop, axis=1)

            if index == 0:
                avg_mcc = result_df.mean()
            else:
                avg_mcc = pd.concat([avg_mcc, result_df.mean()], axis=1)

        except:
            print(file + '.pickle' + " does not exist !")

    avg_mcc.T.to_csv(path_to_saved_csv + "total_avg_mcc.csv")


if __name__ == '__main__':

    path_to_datasets = 'datasets/'
    path_to_saved_file = 'dump/cv_results_dcm/'
    path_to_saved_csv = 'dump/prediction_results_dcm/'
    preprocessing_types = ['default', 'adaptive']

    cv_prediction_using_dcm(path_to_datasets, path_to_saved_file, preprocessing_types)
    get_prediction_results(path_to_datasets, path_to_saved_file, path_to_saved_csv)
    calc_avg_performance(path_to_datasets, path_to_saved_csv)

