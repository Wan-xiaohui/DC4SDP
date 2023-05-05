# Importing problexity
import pandas as pd
import problexity as px
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances
from utils.helper import *
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from src import CFS


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


def get_ih_measures(path_to_datasets, path_to_saved_csvs):
    data_list, label_list, fname = load_data(path_to_datasets)

    for index, file in enumerate(fname):
        csv_file = path_to_saved_csvs + '{}_ih_measures.csv'.format(file)
        result_df = pd.read_csv(csv_file)

        if index == 0:
            total_result_df = copy.deepcopy(result_df)

        else:
            total_result_df = pd.concat([total_result_df, result_df], axis=0)

    return total_result_df
    # corrMatrix = total_result_df.corr(method="spearman").round(3)

    # # sns.set(font_size=2)
    # plt.figure(figsize=(45, 15))
    # mask = np.zeros_like(corrMatrix)
    # mask[np.triu_indices_from(mask)] = True
    # hmap = sns.heatmap(corrMatrix,
    #                    # cmap=sns.diverging_palette(370, 120, n=80, as_cmap=True),
    #                    mask=mask,
    #                    square=True,
    #                    linewidths=0.3,
    #                    cmap="RdBu_r",
    #                    # vmin=-1,
    #                    # vmax=1,
    #                    fmt='.2f',
    #                    annot=True,
    #                    annot_kws={"size": 12}
    #                    )
    # # fontsize can be adjusted to not be giant
    # # hmap.axes.set_title("spearman correlation matrix for the hardness measures", fontsize=20)
    # # labelsize can be adjusted to not be giant
    # hmap.tick_params(labelsize=12)
    #
    # # saves plot output, change'C:/Users/bague/Downloads/cases_correlation.png' to whatever your directory should be and the new filename
    # plt.savefig('output_plots/ih_measures_correlation.pdf', dpi=500)


# def calc_dc_measures(path_to_datasets):
#     data_list, label_list, fname = load_data(path_to_datasets)
#     total_dcm = []
#     for index, file in enumerate(fname):
#         print('\tFile:\t' + file + '...', flush=True)
#
#         try:
#
#             cc = px.ComplexityCalculator()
#             cc.fit(data_list[index], label_list[index])
#             dcm_dic = cc.report()['complexities']
#
#             dcm_lst = [dcm_dic[dcm_name] for dcm_name in dc_measures]
#             # 为了保持和论文中的符号表示一致，这里将字典中所有键的字符串改为首字母大写
#             # dcm_lst = []
#             # for key, values in list(dcm_dic.items()):
#             #     print()
#             #     dcm_lst.append(values)
#             #     # new_key = key.capitalize()
#             #     # if new_key != key:
#             #     #     dcm_dic[new_key] = dcm_dic.pop(key)
#             #     #     if new_key == "Clscoef":
#             #     #         dcm_dic["ClsCoef"] = dcm_dic.pop(new_key)
#             _, ib3 = ft_IBI3_and_BI3(data_list[index], label_list[index])
#             dcm_lst.append(ib3)
#
#         except Exception as e:
#             print('The following error occurred in case of the dataset {}: \n{}'.format(file, e))
#
#         total_dcm.append(dcm_lst)
#
#     return pd.DataFrame(total_dcm)
#
#     #     result_df = pd.read_csv(csv_file)
#     #
#     #     if index == 0:
#     #         total_result_df = copy.deepcopy(result_df)
#     #
#     #     else:
#     #         total_result_df = pd.concat([total_result_df, result_df], axis=0)
#     #
#     # return total_result_df
#     # corrMatrix = total_result_df.corr(method="spearman").round(3)
#
#     # # sns.set(font_size=2)
#     # plt.figure(figsize=(45, 15))
#     # mask = np.zeros_like(corrMatrix)
#     # mask[np.triu_indices_from(mask)] = True
#     # hmap = sns.heatmap(corrMatrix,
#     #                    # cmap=sns.diverging_palette(370, 120, n=80, as_cmap=True),
#     #                    mask=mask,
#     #                    square=True,
#     #                    linewidths=0.3,
#     #                    cmap="RdBu_r",
#     #                    # vmin=-1,
#     #                    # vmax=1,
#     #                    fmt='.2f',
#     #                    annot=True,
#     #                    annot_kws={"size": 12}
#     #                    )
#     # # fontsize can be adjusted to not be giant
#     # # hmap.axes.set_title("spearman correlation matrix for the hardness measures", fontsize=20)
#     # # labelsize can be adjusted to not be giant
#     # hmap.tick_params(labelsize=12)
#     #
#     # # saves plot output, change'C:/Users/bague/Downloads/cases_correlation.png' to whatever your directory should be and the new filename
#     # plt.savefig('output_plots/ih_measures_correlation.pdf', dpi=500)


def calc_dc_measures(path_to_datasets, is_normalized=False, is_feature_selection=False, is_resampling=False):
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

            if is_normalized:
                scaler = preprocessing.StandardScaler().fit(X)
                X = scaler.transform(X)

            if is_feature_selection:
                selected_cols = CFS.cfs(X, y)
                X = X[:, selected_cols]

            if is_resampling:
                smo = SMOTE(random_state=42)
                X, y = smo.fit_resample(X, y)

            cc = px.ComplexityCalculator()
            cc.fit(X, y.values)
            dcm_dic = cc.report()['complexities']

            dcm_lst = [dcm_dic[dcm_name] for dcm_name in dc_measures]
            # dcm_lst = []
            # for key, values in list(dcm_dic.items()):
            #     print()
            #     dcm_lst.append(values)
            #     # new_key = key.capitalize()
            #     # if new_key != key:
            #     #     dcm_dic[new_key] = dcm_dic.pop(key)
            #     #     if new_key == "Clscoef":
            #     #         dcm_dic["ClsCoef"] = dcm_dic.pop(new_key)
            _, ib3 = ft_IBI3_and_BI3(X, y.values)
            dcm_lst.append(ib3)

        except Exception as e:
            print('The following error occurred in case of the dataset {}: \n{}'.format(file, e))

        total_dcm.append(dcm_lst)

    return pd.DataFrame(total_dcm)


if __name__ == '__main__':

    path_to_datasets = 'datasets/'
    path_to_saved_csvs = 'output_csvs/dataset_complexity/'

    # total_dc_df = calc_dc_measures(path_to_datasets)
    # total_dc_df.to_csv(path_to_saved_csvs+"org_dc_measures.csv")

    total_dc_df_1 = calc_dc_measures(path_to_datasets, is_normalized=True)
    total_dc_df_1.to_csv(path_to_saved_csvs+"norm_dc_measures.csv")

    total_dc_df_2 = calc_dc_measures(path_to_datasets, is_normalized=True, is_feature_selection=True)
    total_dc_df_2.to_csv(path_to_saved_csvs + "norm_fs_dc_measures.csv")

    total_dc_df_3 = calc_dc_measures(path_to_datasets, is_normalized=True, is_feature_selection=True, is_resampling=True)
    total_dc_df_3.to_csv(path_to_saved_csvs + "norm_fs_rs_dc_measures.csv")

    # total_hm_df = get_ih_measures(path_to_datasets, path_to_saved_csvs)

#     corrmatrix = []
#     ih_ind_dic, ih_class_dic = calc_ih(model_names, test_truth_dic, test_pred_dic, test_pred_prob_dic)
#
#     ih_vs_hm_corr = calc_ih_hm_correlation(ih_ind_dic, total_hm_df, path_to_datasets)
#     corrmatrix.append(ih_vs_hm_corr.tolist())
#
#     for model_name in model_names:
#         ih_ind_dic, ih_class_dic = calc_ih([model_name], test_truth_dic, test_pred_dic, test_pred_prob_dic)
#         ih_vs_hm_corr = calc_ih_hm_correlation(ih_ind_dic, total_hm_df, path_to_datasets)
#         corrmatrix.append(ih_vs_hm_corr.tolist())
#
#     corrmatrix = pd.DataFrame(corrmatrix)
#     corrmatrix.to_csv("output_csvs/ih_hm_corr.csv")
#
#     # -----------------------------------------------------------------------------------------
#     total_positive_hm_df, total_negative_hm_df = get_partial_ih_measures(path_to_datasets, path_to_saved_csvs)
#
#     positive_corrmatrix = []
#     negative_corrmatrix = []
#     ih_ind_positive_dic, ih_ind_negative_dic, _, _ = calc_partial_ih(
#         model_names, test_truth_dic, test_pred_dic, test_pred_prob_dic)
#
#     positive_ih_vs_hm_corr = calc_ih_hm_correlation(ih_ind_positive_dic, total_positive_hm_df, path_to_datasets)
#     negative_ih_vs_hm_corr = calc_ih_hm_correlation(ih_ind_negative_dic, total_negative_hm_df, path_to_datasets)
#
#     positive_corrmatrix.append(positive_ih_vs_hm_corr.tolist())
#     negative_corrmatrix.append(negative_ih_vs_hm_corr.tolist())
#
#     for model_name in model_names:
#         ih_ind_positive_dic, ih_ind_negative_dic, ih_class_positive_dic, ih_class_negative_dic = calc_partial_ih(
#             [model_name], test_truth_dic, test_pred_dic, test_pred_prob_dic)
#         positive_ih_vs_hm_corr = calc_ih_hm_correlation(ih_ind_positive_dic, total_positive_hm_df, path_to_datasets)
#         negative_ih_vs_hm_corr = calc_ih_hm_correlation(ih_ind_negative_dic, total_negative_hm_df, path_to_datasets)
#         positive_corrmatrix.append(positive_ih_vs_hm_corr.tolist())
#         negative_corrmatrix.append(negative_ih_vs_hm_corr.tolist())
#
#     positive_corrmatrix = pd.DataFrame(positive_corrmatrix)
#     negative_corrmatrix = pd.DataFrame(negative_corrmatrix)
#     positive_corrmatrix.to_csv("output_csvs/positive_ih_hm_corr.csv")
#     negative_corrmatrix.to_csv("output_csvs/negative_ih_hm_corr.csv")
#
#
# # Initialize CoplexityCalculator with default parametrization
# cc = px.ComplexityCalculator()
#
# # Fit model with data
# cc.fit(X, y)
#
# print(cc._metrics())
# print(cc.complexity)
#
# # Prepare figure
# fig = plt.figure(figsize=(7, 7))
#
# # Generate plot describing the dataset
# cc.plot(fig, (1, 1, 1))
# plt.show()

