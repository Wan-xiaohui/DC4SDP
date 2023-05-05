import pandas as pd

from utils.helper import *
from src import instance_hardness
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
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances
from utils.helper import *


class Error(Exception):
    pass

dc_measures = ["f1", "f1v", "f2", "f3", "f4", "l1", "l2", "l3", "n1", "n2", "n3", "n4", "t1",
               "lsc", "density", "clsCoef", "hubs", "t2", "t3", "t4", "c1", "c2"]


def ft_IBI3_and_BI3(data, label, k=5):
    """
    Supplement: Bayes Imbalance Impact Index (BI3). Reference: https://github.com/jasonyanglu/BI3
    :param data: np.ndarray
    :param label: np.ndarray
    :param k: the recommended value is 5
    :return:
    """
    pos_num = sum(label == 1)
    neg_num = sum(label == 0)
    pos_idx = np.nonzero(label == 1)
    neg_idx = np.nonzero(label == 0)
    pos_data = data[pos_idx]
    rr = neg_num / pos_num

    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(data)
    distances, knn_idx = nbrs.kneighbors(pos_data)

    p2 = np.zeros(pos_num)
    p2old = np.zeros(pos_num)
    knn_idx = np.delete(knn_idx, 0, 1)

    for i in range(pos_num):
        p2[i] = np.intersect1d(knn_idx[i], neg_idx).size / k
        p2old[i] = p2[i]

        if p2[i] == 1:
            dist = pairwise_distances(pos_data[i].reshape(1, -1), data).reshape(-1)
            sort_idx = np.argsort(dist)
            nearest_pos = np.nonzero(label[sort_idx] == 1)[0][1]
            p2[i] = (nearest_pos - 1) / nearest_pos

    p1 = 1 - p2

    px = (rr * p1 / (p2 + rr * p1) - p1)

    pm = np.mean(px)

    return px, pm


def calc_dc_measures_norm(path_to_datasets, normalization_type):
    data_list, label_list, fname = load_data(path_to_datasets)
    total_dcm = []
    for index, file in enumerate(fname):
        print('\tFile:\t' + file + '...', flush=True)

        try:
            data = data_list[index]
            label = label_list[index]
            dataset = pd.DataFrame(np.column_stack((data, label)))

            X = dataset.iloc[:, :-1]
            y = dataset.iloc[:, -1]

            if normalization_type == "standard":
                # 数据归一化处理
                scaler = preprocessing.StandardScaler().fit(X)
                X = scaler.transform(X)

            elif normalization_type == "min-max":
                # 数据归一化处理
                scaler = preprocessing.MinMaxScaler().fit(X)
                X = scaler.transform(X)

            else:
                raise Error('Unexpected \'by\' type {}'.format(normalization_type))

            # # 特征选择
            # selected_cols = CFS.cfs(X, y)
            # X = X[:, selected_cols]
            #
            # # SMOTE过采样
            # smo = SMOTE(random_state=42)
            # X, y = smo.fit_resample(X, y)

            cc = px.ComplexityCalculator()
            cc.fit(X, y.values)
            dcm_dic = cc.report()['complexities']

            dcm_lst = [dcm_dic[dcm_name] for dcm_name in dc_measures]
            _, ib3 = ft_IBI3_and_BI3(X, y.values)
            dcm_lst.append(ib3)

        except Exception as e:
            print('The following error occurred in case of the dataset {}: \n{}'.format(file, e))

        total_dcm.append(dcm_lst)

    return pd.DataFrame(total_dcm)


def calc_dc_measures_fs(path_to_datasets, fs_type):
    data_list, label_list, fname = load_data(path_to_datasets)
    total_dcm = []
    for index, file in enumerate(fname):
        print('\tFile:\t' + file + '...', flush=True)

        try:
            data = data_list[index]
            label = label_list[index]
            dataset = pd.DataFrame(np.column_stack((data, label)))

            X = dataset.iloc[:, :-1]
            y = dataset.iloc[:, -1]

            # 数据归一化处理
            scaler = preprocessing.StandardScaler().fit(X)
            X = scaler.transform(X)

            # 特征选择
            if fs_type == "CFS":
                selected_cols = CFS.cfs(X, y)

            elif fs_type == "SKB_anova":
                n_features = int(0.7 * X.shape[1])
                selector = SelectKBest(f_classif, k=n_features)
                selector.fit(X, y)
                selected_cols = selector.get_support(indices=True)

            elif fs_type == "SKB_mutual":
                n_features = int(0.7 * X.shape[1])
                selector = SelectKBest(mutual_info_classif, k=n_features)
                selector.fit(X, y)
                selected_cols = selector.get_support(indices=True)

            elif fs_type == "ReliefF":
                n_features = int(0.7 * X.shape[1])
                selector = ReliefF(n_features_to_keep=n_features)
                selector.fit(X, y)
                selected_cols = selector.top_features[0:selector.n_features_to_keep]

            elif fs_type == "LinSVM":
                clf = LinearSVC()
                clf.fit(X, y)
                model = SelectFromModel(clf, prefit=True)
                selected_cols = model.get_support(indices=True)

            elif fs_type == "Tree":
                clf = ExtraTreesClassifier()
                clf.fit(X, y)
                model = SelectFromModel(clf, prefit=True)
                selected_cols = model.get_support(indices=True)

            else:
                raise Error('Unexpected \'by\' type {}'.format(fs_type))

            X = X[:, selected_cols]

            # # SMOTE过采样
            # smo = SMOTE(random_state=42)
            # X, y = smo.fit_resample(X, y)

            cc = px.ComplexityCalculator()
            cc.fit(X, y.values)
            dcm_dic = cc.report()['complexities']

            dcm_lst = [dcm_dic[dcm_name] for dcm_name in dc_measures]
            _, ib3 = ft_IBI3_and_BI3(X, y.values)
            dcm_lst.append(ib3)

        except Exception as e:
            print('The following error occurred in case of the dataset {}: \n{}'.format(file, e))

        total_dcm.append(dcm_lst)

    return pd.DataFrame(total_dcm)


def calc_dc_measures_rs(path_to_datasets, rs_type):
    data_list, label_list, fname = load_data(path_to_datasets)
    total_dcm = []
    for index, file in enumerate(fname):
        print('\tFile:\t' + file + '...', flush=True)

        try:
            data = data_list[index]
            label = label_list[index]
            dataset = pd.DataFrame(np.column_stack((data, label)))

            X = dataset.iloc[:, :-1]
            y = dataset.iloc[:, -1]

            # 数据归一化处理
            scaler = preprocessing.StandardScaler().fit(X)
            X = scaler.transform(X)

            # 特征选择
            selected_cols = CFS.cfs(X, y)
            X = X[:, selected_cols]

            # 重采样
            if rs_type == "SMOTE":
                sampler = SMOTE(random_state=42)
                X, y = sampler.fit_resample(X, y)

            elif rs_type == "ADASYN":
                sampler = ADASYN(random_state=42)
                X, y = sampler.fit_resample(X, y)

            elif rs_type == "BorderSMOTE":
                sampler = BorderlineSMOTE(random_state=42)
                X, y = sampler.fit_resample(X, y)

            elif rs_type == "SMOTETomek":
                sampler = SMOTETomek(random_state=42)
                X, y = sampler.fit_resample(X, y)

            elif rs_type == "SMOTEENN":
                sampler = SMOTEENN(random_state=42)
                X, y = sampler.fit_resample(X, y)

            elif rs_type == "RUS":
                sampler = RandomUnderSampler(random_state=42)
                X, y = sampler.fit_resample(X, y)

            else:
                raise Error('Unexpected \'by\' type {}'.format(rs_type))

            cc = px.ComplexityCalculator()
            cc.fit(X, y.values)
            dcm_dic = cc.report()['complexities']

            dcm_lst = [dcm_dic[dcm_name] for dcm_name in dc_measures]
            _, ib3 = ft_IBI3_and_BI3(X, y.values)
            dcm_lst.append(ib3)

        except Exception as e:
            print('The following error occurred in case of the dataset {}: \n{}'.format(file, e))

        total_dcm.append(dcm_lst)

    return pd.DataFrame(total_dcm)


if __name__ == '__main__':

    path_to_datasets = 'datasets/'
    path_to_saved_csvs = 'output_csvs/dataset_complexity/'

    data_normalizations = ["standard", "min-max"]
    for normalization_type in data_normalizations:
        total_dc_df = calc_dc_measures_norm(path_to_datasets, normalization_type)
        total_dc_df.to_csv(path_to_saved_csvs + "{}_norm_dc_measures.csv".format(normalization_type))

    feature_selections = ["CFS", "SKB_anova", "SKB_mutual", "ReliefF", "LinSVM", "Tree"]
    for fs_type in feature_selections:
        total_dc_df = calc_dc_measures_fs(path_to_datasets, fs_type)
        total_dc_df.to_csv(path_to_saved_csvs + "{}_fs_dc_measures.csv".format(fs_type))

    data_resamplings = ["SMOTE", "ADASYN", "BorderSMOTE", "SMOTETomek", "SMOTEENN", "RUS"]
    for rs_type in data_resamplings:
        total_dc_df = calc_dc_measures_rs(path_to_datasets, rs_type)
        total_dc_df.to_csv(path_to_saved_csvs + "{}_rs_dc_measures.csv".format(rs_type))