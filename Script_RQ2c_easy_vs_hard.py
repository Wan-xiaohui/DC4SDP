from utils.helper import *
from src import instance_hardness
import numpy as np
import pickle
import pandas as pd
from scipy.stats import spearmanr


def cv_prediction(path_to_datasets, path_to_saved_file, model_names):

    data_list, label_list, fname = load_data(path_to_datasets)

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


def calc_ih(model_names, test_truth_dic, test_pred_dic, test_pred_prob_dic):
    dataset_names = test_truth_dic.keys()
    ih_ind_dic = {}
    ih_class_dic = {}
    for dataset_name in dataset_names:
        dataset_ih_ind = []
        dataset_ih_class = []

        labels = test_truth_dic[dataset_name]
        test_preds = test_pred_dic[dataset_name]
        test_pred_probs = test_pred_prob_dic[dataset_name]
        n_samples = len(labels)

        for i in range(n_samples):
            one_sample_preds = []
            one_sample_pred_probs = []

            for clf in model_names:
                one_sample_preds.extend(test_preds[i][clf])
                one_sample_pred_probs.extend(test_pred_probs[i][clf])

            one_sample_preds = np.array(one_sample_preds).astype(np.int)
            one_sample_pred_probs = np.array(one_sample_pred_probs).astype(np.float)

            one_sample_label = np.repeat(labels[i], len(model_names)*5).astype(np.int)
            ih_ind = np.mean(one_sample_preds != one_sample_label)
            dataset_ih_ind.append(ih_ind)

            ih_class_1 = one_sample_pred_probs * (one_sample_preds != one_sample_label)
            ih_class_2 = (np.ones_like(one_sample_pred_probs) - one_sample_pred_probs) * (one_sample_preds == one_sample_label)
            ih_class = np.mean(ih_class_1 + ih_class_2)
            dataset_ih_class.append(ih_class)

        ih_ind_dic[dataset_name] = dataset_ih_ind
        ih_class_dic[dataset_name] = dataset_ih_class

    return ih_ind_dic, ih_class_dic


def calc_partial_ih(model_names, test_truth_dic, test_pred_dic, test_pred_prob_dic):
    dataset_names = test_truth_dic.keys()
    ih_ind_positive_dic = {}
    ih_ind_negative_dic = {}
    ih_class_positive_dic = {}
    ih_class_negative_dic = {}

    for dataset_name in dataset_names:
        dataset_ih_ind_positive = []
        dataset_ih_ind_negative = []
        dataset_ih_class_positive = []
        dataset_ih_class_negative = []

        labels = test_truth_dic[dataset_name]
        test_preds = test_pred_dic[dataset_name]
        test_pred_probs = test_pred_prob_dic[dataset_name]

        positive_inds = np.where(np.array(labels).astype(int) == 1)[0].tolist()
        negative_inds = np.where(np.array(labels).astype(int) == 0)[0].tolist()

        for i in positive_inds:
            one_sample_preds = []
            one_sample_pred_probs = []

            for clf in model_names:
                one_sample_preds.extend(test_preds[i][clf])
                one_sample_pred_probs.extend(test_pred_probs[i][clf])

            one_sample_preds = np.array(one_sample_preds).astype(np.int)
            one_sample_pred_probs = np.array(one_sample_pred_probs).astype(np.float)

            one_sample_label = np.repeat(labels[i], len(model_names) * 5).astype(np.int)
            ih_ind = np.mean(one_sample_preds != one_sample_label)
            dataset_ih_ind_positive.append(ih_ind)

            ih_class_1 = one_sample_pred_probs * (one_sample_preds != one_sample_label)
            ih_class_2 = (np.ones_like(one_sample_pred_probs) - one_sample_pred_probs) * (
                        one_sample_preds == one_sample_label)
            ih_class = np.mean(ih_class_1 + ih_class_2)
            dataset_ih_class_positive.append(ih_class)

        ih_ind_positive_dic[dataset_name] = dataset_ih_ind_positive
        ih_class_positive_dic[dataset_name] = dataset_ih_class_positive

        for i in negative_inds:
            one_sample_preds = []
            one_sample_pred_probs = []

            for clf in model_names:
                one_sample_preds.extend(test_preds[i][clf])
                one_sample_pred_probs.extend(test_pred_probs[i][clf])

            one_sample_preds = np.array(one_sample_preds).astype(np.int)
            one_sample_pred_probs = np.array(one_sample_pred_probs).astype(np.float)

            one_sample_label = np.repeat(labels[i], len(model_names) * 5).astype(np.int)
            ih_ind = np.mean(one_sample_preds != one_sample_label)
            dataset_ih_ind_negative.append(ih_ind)

            ih_class_1 = one_sample_pred_probs * (one_sample_preds != one_sample_label)
            ih_class_2 = (np.ones_like(one_sample_pred_probs) - one_sample_pred_probs) * (
                        one_sample_preds == one_sample_label)
            ih_class = np.mean(ih_class_1 + ih_class_2)
            dataset_ih_class_negative.append(ih_class)

        ih_ind_negative_dic[dataset_name] = dataset_ih_ind_negative
        ih_class_negative_dic[dataset_name] = dataset_ih_class_negative

    return ih_ind_positive_dic, ih_ind_negative_dic, ih_class_positive_dic, ih_class_negative_dic


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


def get_partial_ih_measures(path_to_datasets, path_to_saved_csvs):
    data_list, label_list, fname = load_data(path_to_datasets)

    for index, file in enumerate(fname):
        csv_file = path_to_saved_csvs + '{}_ih_measures.csv'.format(file)
        result_df = pd.read_csv(csv_file)
        positive_result_df = result_df.loc[np.where(label_list[index] == 1)].reset_index(drop=True)
        negative_result_df = result_df.loc[np.where(label_list[index] == 0)].reset_index(drop=True)

        if index == 0:
            # total_result_df = copy.deepcopy(result_df)
            total_positive_result_df = copy.deepcopy(positive_result_df)
            total_negative_result_df = copy.deepcopy(negative_result_df)

        else:
            # total_result_df = pd.concat([total_result_df, result_df], axis=0)
            total_positive_result_df = pd.concat([total_positive_result_df, positive_result_df], axis=0)
            total_negative_result_df = pd.concat([total_negative_result_df, negative_result_df], axis=0)

    return total_positive_result_df, total_negative_result_df


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
        # p = pval.applymap(lambda x: ''.join(['*' for t in [.05, .01, .001] if x <= t]))
        p = pval.applymap(lambda x: '*' if x <= 0.05 else ' ')

        rho_lst.append(rho.iloc[0, 1:])
        p_lst.append(p.iloc[0, 1:])

    return rho_lst, p_lst


def calc_easy_hard_instance(ih_dic, path_to_datasets, total_hm_df):

    _, _, fname = load_data(path_to_datasets)

    for index, file in enumerate(fname):
        ih_df = pd.DataFrame(ih_dic[file])

        if index == 0:
            total_ih_df = copy.deepcopy(ih_df)

        else:
            total_ih_df = pd.concat([total_ih_df, ih_df], axis=0)

    total_ih_df.reset_index(drop=True, inplace=True)
    hard_idx = total_ih_df.loc[total_ih_df[0] >= 0.7].index.to_list()
    easy_idx = total_ih_df.loc[total_ih_df[0] <= 0.3].index.to_list()

    hard_ih_measures = total_hm_df.iloc[hard_idx]
    easy_ih_measures = total_hm_df.iloc[easy_idx]

    return hard_ih_measures, easy_ih_measures


if __name__ == '__main__':

    model_names = ['KNN', 'NB', 'CART', 'LR', 'SVM', 'MLP',
                   'Boosted_RS', 'Greedy_RL', 'RF', 'SVMBoosting', 'MLPBoosting']

    path_to_datasets = 'datasets/'

    path_to_saved_file = 'dump/prediction_results/'
    test_truth_dic, test_pred_dic, test_pred_prob_dic = get_prediction_results(
        path_to_datasets, path_to_saved_file)

    # -----------------------------------------------------------------------------------------
    path_to_saved_csvs = 'output_csvs/hardness_measures/'
    total_positive_result_df, total_negative_result_df = get_partial_ih_measures(path_to_datasets, path_to_saved_csvs)
    total_positive_result_df.reset_index(drop=True, inplace=True)
    total_negative_result_df.reset_index(drop=True, inplace=True)

    positive_avg = total_positive_result_df.mean()
    positive_max = total_positive_result_df.max()
    positive_std = total_positive_result_df.std()
    negative_avg = total_negative_result_df.mean()
    negative_max = total_negative_result_df.max()
    negative_std = total_negative_result_df.std()

    total_hm_df = get_ih_measures(path_to_datasets, path_to_saved_csvs)
    total_hm_df.reset_index(drop=True, inplace=True)
    ih_dic, _ = calc_ih(model_names, test_truth_dic, test_pred_dic, test_pred_prob_dic)
    hard_ih_measures, easy_ih_measures = calc_easy_hard_instance(ih_dic, path_to_datasets, total_hm_df)

    hard_avg = hard_ih_measures.mean()
    hard_max = hard_ih_measures.max()
    hard_std = hard_ih_measures.std()

    easy_avg = easy_ih_measures.mean()
    easy_max = easy_ih_measures.max()
    easy_std = easy_ih_measures.std()

    print()