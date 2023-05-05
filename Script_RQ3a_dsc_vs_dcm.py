import copy
from utils.helper import *
from src import instance_hardness
import numpy as np
import pickle
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import matthews_corrcoef
import problexity as px
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances
from utils.helper import *


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


def calc_performance(label_true, label_pred):
    MCC = matthews_corrcoef(label_true, label_pred)
    return MCC


def calc_dc_measures(path_to_datasets):
    data_list, label_list, fname = load_data(path_to_datasets)
    total_dcm = []
    for index, file in enumerate(fname):
        print('\tFile:\t' + file + '...', flush=True)

        try:

            cc = px.ComplexityCalculator()
            cc.fit(data_list[index], label_list[index])
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
            _, ib3 = ft_IBI3_and_BI3(data_list[index], label_list[index])
            dcm_lst.append(ib3)

        except Exception as e:
            print('The following error occurred in case of the dataset {}: \n{}'.format(file, e))

        total_dcm.append(dcm_lst)

    return pd.DataFrame(total_dcm)

    #     result_df = pd.read_csv(csv_file)
    #
    #     if index == 0:
    #         total_result_df = copy.deepcopy(result_df)
    #
    #     else:
    #         total_result_df = pd.concat([total_result_df, result_df], axis=0)
    #
    # return total_result_df
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


def cv_prediction(path_to_datasets, path_to_saved_file, start, end, model_names):

    data_list, label_list, fname = load_data(path_to_datasets)

    data_list = data_list[start-1:end-1]
    label_list = label_list[start-1:end-1]
    fname = fname[start-1:end-1]

    for index, file in enumerate(fname):

        print('File:\t' + file + '...', flush=True)

        try:
            data = data_list[index]
            label = label_list[index]

            dataset = pd.DataFrame(np.column_stack((data, label)))

            X = dataset.iloc[:, :-1]
            y = dataset.iloc[:, -1]

            test_truths, test_preds, test_pred_probs = instance_hardness.repeated_cross_validation_Opt_Clfs(X, y, model_names)

            pkfile = open(path_to_saved_file + file + '.pickle', 'wb')

            pickle.dump(test_truths, pkfile)
            pickle.dump(test_preds, pkfile)
            pickle.dump(test_pred_probs, pkfile)

        except Exception as e:
            print('The following error occurred in case of the dataset {}: \n{}'.format(file, e))


def get_prediction_results(path_to_datasets, path_to_saved_file):
    _, _, fname = load_data(path_to_datasets)

    test_truth_dic, test_pred_dic, test_pred_prob_dic = {}, {}, {}
    for index, file in enumerate(fname):
        pkfile = open(path_to_saved_file + file + '.pickle', 'rb')
        test_truth_dic[file] = pickle.load(pkfile)
        test_pred_dic[file] = pickle.load(pkfile)
        test_pred_prob_dic[file] = pickle.load(pkfile)

    return test_truth_dic, test_pred_dic, test_pred_prob_dic


def calc_dataset_mcc(test_truth_dic, test_pred_dic):
    avg_mcc_lst = []
    clf_names = ['KNN', 'NB', 'CART', 'LR', 'SVM', 'MLP', 'Boosted_RS', 'Greedy_RL', 'RF', 'SVMBoosting', 'MLPBoosting']

    for dataset_name, dataset_clf_pred_dic in test_pred_dic.items():
        clf_mcc_lst = []
        for i in range(5):
            for clf_name in clf_names:
                clf_pred_lst = []
                for j in range(len(dataset_clf_pred_dic)):
                    clf_pred_lst.append(dataset_clf_pred_dic[j][clf_name][i])
                y_pred = np.array(clf_pred_lst)
                y_true = np.array(test_truth_dic[dataset_name])
                mcc_value = calc_performance(y_true, y_pred)
                clf_mcc_lst.append(mcc_value)

        avg_mcc = np.mean(clf_mcc_lst)
        avg_mcc_lst.append(avg_mcc)
    return np.array(avg_mcc_lst)


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


def get_dataset_ih_measures(path_to_datasets, path_to_saved_csvs):
    data_list, label_list, fname = load_data(path_to_datasets)
    ih_measure_dic = {}

    for index, file in enumerate(fname):
        csv_file = path_to_saved_csvs + '{}_ih_measures.csv'.format(file)
        result_df = pd.read_csv(csv_file)
        ih_measure_dic[file] = result_df

    return ih_measure_dic


def calc_ih_hm_correlation(ih_dic, total_hm_df, path_to_datasets):
    _, _, fname = load_data(path_to_datasets)

    for index, file in enumerate(fname):
        ih_df = pd.DataFrame(ih_dic[file])

        if index == 0:
            total_ih_df = copy.deepcopy(ih_df)

        else:
            total_ih_df = pd.concat([total_ih_df, ih_df], axis=0)

    total_result_df = pd.concat([total_ih_df, total_hm_df], axis=1)

    rho = total_result_df.corr(method="spearman").round(3)
    pval = total_result_df.corr(method=lambda x, y: spearmanr(x, y)[1]) - np.eye(*rho.shape)
    p = pval.applymap(lambda x: ''.join(['*' for t in [.05, .01, .001] if x <= t]))
    # rho.round(3).astype(str) + p

    # corrMatrix = total_result_df.corr(method="spearman").round(3)

    return rho.iloc[0, 1:], p.iloc[0, 1:]


def calc_dataset_ih_hm_correlation(ih_dic, hm_dic, path_to_datasets):
    _, _, fname = load_data(path_to_datasets)
    rho_lst = []
    p_lst = []

    for index, file in enumerate(fname):
        ih_df = pd.DataFrame(ih_dic[file])
        hm_df = pd.DataFrame(hm_dic[file])

        total_result_df = pd.concat([ih_df, hm_df], axis=1)

        rho = total_result_df.corr(method="spearman")
        pval = total_result_df.corr(method=lambda x, y: spearmanr(x, y)[1]) - np.eye(*rho.shape)
        p = pval.applymap(lambda x: ''.join(['*' for t in [.05, .01, .001] if x <= t]))

        rho_lst.append(rho.iloc[0, 1:])
        p_lst.append(p.iloc[0, 1:])

    return rho_lst, p_lst


def calc_dh_dcm_correlation(dataset_hardness_df, dcm_df, path_to_datasets):
    _, _, fname = load_data(path_to_datasets)
    total_result_df = pd.concat([dataset_hardness_df, dcm_df], axis=1)
    rho = total_result_df.corr(method="spearman")
    pval = total_result_df.corr(method=lambda x, y: spearmanr(x, y)[1]) - np.eye(*rho.shape)
    p = pval.applymap(lambda x: ''.join(['*' for t in [.05, .01, .001] if x <= t]))
    rho = rho.iloc[0, 1:]
    p = p.iloc[0, 1:]

    return rho, p


def calc_dc_measures_correlation(path_to_saved_csvs):

    result_df = pd.read_csv(path_to_saved_csvs, index_col=0)
    result_df.columns = ["F1", "F1v", "F2", "F3", "F4", "L1", "L2", "L3", "N1", "N2", "N3", "N4",
                         "T1", "Lsc", "Density", "ClsCoef", "Hubs", "T2", "T3", "T4", "C1", "C2", "BI3"]
    corrMatrix = result_df.corr(method="spearman").round(3)

    # sns.set(font_size=2)
    plt.figure(figsize=(45, 15))
    mask = np.zeros_like(corrMatrix)
    mask[np.triu_indices_from(mask)] = True
    hmap = sns.heatmap(corrMatrix,
                       # cmap=sns.diverging_palette(370, 120, n=80, as_cmap=True),
                       mask=mask,
                       square=True,
                       linewidths=0.3,
                       cmap="RdBu_r",
                       # vmin=-1,
                       # vmax=1,
                       fmt='.3f',
                       annot=True,
                       annot_kws={"size": 10}
                       )
    # fontsize can be adjusted to not be giant
    # hmap.axes.set_title("spearman correlation matrix for the hardness measures", fontsize=20)
    # labelsize can be adjusted to not be giant
    hmap.tick_params(labelsize=12)

    # saves plot output, change'C:/Users/bague/Downloads/cases_correlation.png' to whatever your directory should be and the new filename
    plt.savefig('output_plots/dc_measures_correlation.pdf', dpi=500)


if __name__ == '__main__':

    model_names = ['KNN', 'NB', 'CART', 'LR', 'SVM', 'MLP',
                   'Boosted_RS', 'Greedy_RL', 'RF', 'SVMBoosting', 'MLPBoosting']

    path_to_datasets = 'datasets/'

    # path_to_saved_file = 'dump/prediction_results/'
    # test_truth_dic, test_pred_dic, test_pred_prob_dic = get_prediction_results(path_to_datasets, path_to_saved_file)
    #
    # avg_mcc = calc_dataset_mcc(test_truth_dic, test_pred_dic)
    # avg_mcc = np.array(avg_mcc)
    # avg_complexity = 1 - avg_mcc
    # dataset_complexity_df = pd.DataFrame(avg_complexity)
    # dataset_complexity_df.columns = ["imbalanced_dataset_hardness"]
    #
    path_to_saved_csvs = 'output_csvs/dataset_complexity/'
    # dataset_complexity_df.to_csv(path_to_saved_csvs + "imbalanced_dataset_hardness.csv", index=False)
    #
    # total_dc_df = calc_dc_measures(path_to_datasets)
    # total_dc_df.to_csv(path_to_saved_csvs+"org_dc_measures.csv")

    calc_dc_measures_correlation(path_to_saved_csvs+"org_dc_measures.csv")

    # dcm_df = pd.read_csv(path_to_saved_csvs+"org_dc_measures.csv", index_col=0)
    # dcm_df.columns = ["F1", "F1v", "F2", "F3", "F4", "L1", "L2", "L3", "N1", "N2", "N3", "N4",
    #                   "T1", "Lsc", "Density", "ClsCoef", "Hubs", "T2", "T3", "T4", "C1", "C2", "BI3"]
    #
    # new_dcm_df = dcm_df.drop(columns=['F2', 'N3', 'N4', 'Hubs', 'C2'])
    # dh_vs_dcm_rho, dh_vs_dcm_pval = calc_dh_dcm_correlation(dataset_complexity_df, new_dcm_df, path_to_datasets)
    # dh_vs_dcm_rho.to_csv("output_csvs/dsc_dcm_rho.csv")
    # dh_vs_dcm_pval.to_csv("output_csvs/dsc_dcm_pval.csv")

    # path_to_saved_csvs = 'output_csvs/hardness_measures/'
    # dataset_hm_dic = get_dataset_ih_measures(path_to_datasets, path_to_saved_csvs)

    # rho_matrix = []
    # pval_matrix = []
    # ih_ind_dic, ih_class_dic = calc_ih(model_names, test_truth_dic, test_pred_dic, test_pred_prob_dic)
    # avg_ih_lst = []
    # for dataset_name, ih_lst in ih_ind_dic.items():
    #     avg_ih_lst.append(sum(ih_lst)/len(ih_lst))
    #
    # ih_vs_hm_rho, ih_vs_hm_pval = calc_dataset_ih_hm_correlation(ih_ind_dic, dataset_hm_dic, path_to_datasets)
    #
    # ih_vs_hm_rho_df = pd.DataFrame(ih_vs_hm_rho).reset_index(drop=True)
    # ih_vs_hm_rho_df.insert(0, 'avg.ih', avg_ih_lst)
    #
    # ih_vs_hm_pval_df = pd.DataFrame(ih_vs_hm_pval).reset_index(drop=True)
    #
    # ih_vs_hm_rho_df.to_csv("output_csvs/dataset_ih_hm_rho.csv")
    # ih_vs_hm_pval_df.to_csv("output_csvs/dataset_ih_hm_pval.csv")
    #
    # # positive_pval_matrix.to_csv("output_csvs/positive_ih_hm_pval.csv")
    # # negative_rho_matrix.to_csv("output_csvs/negative_ih_hm_rho.csv")
    # # negative_pval_matrix.to_csv("output_csvs/negative_ih_hm_pval.csv")
    #
    # print()
    # rho_matrix.append(ih_vs_hm_rho.tolist())
    # pval_matrix.append(ih_vs_hm_pval.tolist())
    #
    # # 计算单个个模型下的instance hardness
    # for model_name in model_names:
    #     ih_ind_dic, ih_class_dic = calc_ih([model_name], test_truth_dic, test_pred_dic, test_pred_prob_dic)
    #     ih_vs_hm_rho, ih_vs_hm_pval = calc_ih_hm_correlation(ih_ind_dic, total_hm_df, path_to_datasets)
    #     rho_matrix.append(ih_vs_hm_rho)
    #     pval_matrix.append(ih_vs_hm_pval)
    #
    # rho_matrix = pd.DataFrame(rho_matrix).reset_index(drop=True)
    # pval_matrix = pd.DataFrame(pval_matrix).reset_index(drop=True)
    #
    # rho_matrix.to_csv("output_csvs/ih_hm_rho.csv")
    # pval_matrix.to_csv("output_csvs/ih_hm_rho.csv")
    #
    # # -----------------------------------------------------------------------------------------
    # total_positive_hm_df, total_negative_hm_df = get_partial_ih_measures(path_to_datasets, path_to_saved_csvs)
    #
    # positive_rho_matrix = []
    # negative_rho_matrix = []
    # positive_pval_matrix = []
    # negative_pval_matrix = []
    #
    # ih_ind_positive_dic, ih_ind_negative_dic, _, _ = calc_partial_ih(
    #     model_names, test_truth_dic, test_pred_dic, test_pred_prob_dic)
    #
    # positive_ih_vs_hm_rho, positive_ih_vs_hm_pval = calc_ih_hm_correlation(
    #     ih_ind_positive_dic, total_positive_hm_df, path_to_datasets)
    #
    # negative_ih_vs_hm_rho, negative_ih_vs_hm_pval = calc_ih_hm_correlation(
    #     ih_ind_negative_dic, total_negative_hm_df, path_to_datasets)
    #
    # positive_rho_matrix.append(positive_ih_vs_hm_rho)
    # positive_pval_matrix.append(positive_ih_vs_hm_pval)
    #
    # negative_rho_matrix.append(negative_ih_vs_hm_rho)
    # negative_pval_matrix.append(negative_ih_vs_hm_pval)
    #
    # for model_name in model_names:
    #     ih_ind_positive_dic, ih_ind_negative_dic, ih_class_positive_dic, ih_class_negative_dic = calc_partial_ih(
    #         [model_name], test_truth_dic, test_pred_dic, test_pred_prob_dic)
    #
    #     positive_ih_vs_hm_rho, positive_ih_vs_hm_pval = calc_ih_hm_correlation(
    #         ih_ind_positive_dic, total_positive_hm_df, path_to_datasets)
    #
    #     negative_ih_vs_hm_rho, negative_ih_vs_hm_pval = calc_ih_hm_correlation(
    #         ih_ind_negative_dic, total_negative_hm_df, path_to_datasets)
    #
    #     positive_rho_matrix.append(positive_ih_vs_hm_rho)
    #     positive_pval_matrix.append(positive_ih_vs_hm_pval)
    #     negative_rho_matrix.append(negative_ih_vs_hm_rho)
    #     negative_pval_matrix.append(negative_ih_vs_hm_pval)
    #
    # positive_rho_matrix = pd.DataFrame(positive_rho_matrix).reset_index(drop=True)
    # positive_pval_matrix = pd.DataFrame(positive_pval_matrix).reset_index(drop=True)
    # negative_rho_matrix = pd.DataFrame(negative_rho_matrix).reset_index(drop=True)
    # negative_pval_matrix = pd.DataFrame(negative_pval_matrix).reset_index(drop=True)
    #
    # positive_rho_matrix.to_csv("output_csvs/positive_ih_hm_rho.csv")
    # positive_pval_matrix.to_csv("output_csvs/positive_ih_hm_pval.csv")
    # negative_rho_matrix.to_csv("output_csvs/negative_ih_hm_rho.csv")
    # negative_pval_matrix.to_csv("output_csvs/negative_ih_hm_pval.csv")
