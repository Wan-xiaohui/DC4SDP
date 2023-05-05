from utils.helper import *
from src import instance_hardness
import numpy as np
import pickle
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression


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


def calc_lin_ih_hm_corr(ih_dic, total_hm_df, path_to_datasets):
    _, _, fname = load_data(path_to_datasets)

    for index, file in enumerate(fname):
        ih_df = pd.DataFrame(ih_dic[file])

        if index == 0:
            total_ih_df = copy.deepcopy(ih_df)

        else:
            total_ih_df = pd.concat([total_ih_df, ih_df], axis=0)

    total_ih_df.reset_index(drop=True, inplace=True)
    total_ih_df.columns = ['ih']
    total_hm_df.fillna(0, inplace=True)
    total_hm_df.reset_index(drop=True, inplace=True)
    linear_model = LinearRegression()
    linear_model.fit(total_hm_df, total_ih_df)
    new_hm = np.matmul(total_hm_df.values, linear_model.coef_.T)
    total_hm_df = pd.concat([total_hm_df, pd.DataFrame(new_hm, columns=["Lin"]).reset_index(drop=True)], axis=1)

    total_result_df = pd.concat([total_ih_df, total_hm_df], axis=1)

    rho = total_result_df.corr(method="spearman").round(3)
    pval = total_result_df.corr(method=lambda x, y: spearmanr(x, y)[1]) - np.eye(*rho.shape)
    # p = pval.applymap(lambda x: ''.join(['*' for t in [.05, .01, .001] if x <= t]))
    p = pval.applymap(lambda x: '*' if x <= 0.05 else ' ')
    # corrMatrix = total_result_df.corr(method="spearman").round(3)
    return rho.iloc[0, 1:], p.iloc[0, 1:]


if __name__ == '__main__':

    model_names = ['KNN', 'NB', 'CART', 'LR', 'SVM', 'MLP',
                   'Boosted_RS', 'Greedy_RL', 'RF', 'SVMBoosting', 'MLPBoosting']

    path_to_datasets = 'datasets/'

    # Read the test result of all models
    path_to_saved_file = 'dump/prediction_results/'
    test_truth_dic, test_pred_dic, test_pred_prob_dic = get_prediction_results(
        path_to_datasets, path_to_saved_file)

    # -----------------------------------------------------------------------------------------
    # Read hardness measure values in all datasets
    path_to_saved_csvs = 'output_csvs/hardness_measures/'
    total_hm_df = get_ih_measures(path_to_datasets, path_to_saved_csvs)

    # Calculate the instance hardness
    rho_matrix = []
    pval_matrix = []
    ih_ind_dic, ih_class_dic = calc_ih(model_names, test_truth_dic, test_pred_dic, test_pred_prob_dic)

    ih_vs_hm_rho, ih_vs_hm_pval = calc_lin_ih_hm_corr(ih_ind_dic, total_hm_df, path_to_datasets)

    rho_matrix.append(ih_vs_hm_rho.tolist())
    pval_matrix.append(ih_vs_hm_pval.tolist())

    # Calculate the instance hardness for an individual model
    for model_name in model_names:
        ih_ind_dic, ih_class_dic = calc_ih([model_name], test_truth_dic, test_pred_dic, test_pred_prob_dic)
        ih_vs_hm_rho, ih_vs_hm_pval = calc_lin_ih_hm_corr(ih_ind_dic, total_hm_df, path_to_datasets)
        rho_matrix.append(ih_vs_hm_rho)
        pval_matrix.append(ih_vs_hm_pval)

    rho_matrix = pd.DataFrame(rho_matrix).reset_index(drop=True)
    pval_matrix = pd.DataFrame(pval_matrix).reset_index(drop=True)

    rho_matrix.to_csv("output_csvs/lin_ih_hm_rho.csv")
    pval_matrix.to_csv("output_csvs/lin_ih_hm_pval.csv")

    # -----------------------------------------------------------------------------------------
    # Read the hardness measure of majority and minority classes from all datasets
    total_positive_hm_df, total_negative_hm_df = get_partial_ih_measures(path_to_datasets, path_to_saved_csvs)

    # Calculate the hardness of majority and minority class instances for all models.
    positive_rho_matrix = []
    negative_rho_matrix = []
    positive_pval_matrix = []
    negative_pval_matrix = []

    ih_ind_positive_dic, ih_ind_negative_dic, _, _ = calc_partial_ih(
        model_names, test_truth_dic, test_pred_dic, test_pred_prob_dic)

    positive_ih_vs_hm_rho, positive_ih_vs_hm_pval = calc_lin_ih_hm_corr(
        ih_ind_positive_dic, total_positive_hm_df, path_to_datasets)

    negative_ih_vs_hm_rho, negative_ih_vs_hm_pval = calc_lin_ih_hm_corr(
        ih_ind_negative_dic, total_negative_hm_df, path_to_datasets)

    positive_rho_matrix.append(positive_ih_vs_hm_rho)
    positive_pval_matrix.append(positive_ih_vs_hm_pval)

    negative_rho_matrix.append(negative_ih_vs_hm_rho)
    negative_pval_matrix.append(negative_ih_vs_hm_pval)

    # Calculate the instance hardness of majority and minority class samples for each individual model
    for model_name in model_names:
        ih_ind_positive_dic, ih_ind_negative_dic, ih_class_positive_dic, ih_class_negative_dic = calc_partial_ih(
            [model_name], test_truth_dic, test_pred_dic, test_pred_prob_dic)

        positive_ih_vs_hm_rho, positive_ih_vs_hm_pval = calc_lin_ih_hm_corr(
            ih_ind_positive_dic, total_positive_hm_df, path_to_datasets)

        negative_ih_vs_hm_rho, negative_ih_vs_hm_pval = calc_lin_ih_hm_corr(
            ih_ind_negative_dic, total_negative_hm_df, path_to_datasets)

        positive_rho_matrix.append(positive_ih_vs_hm_rho)
        positive_pval_matrix.append(positive_ih_vs_hm_pval)
        negative_rho_matrix.append(negative_ih_vs_hm_rho)
        negative_pval_matrix.append(negative_ih_vs_hm_pval)

    positive_rho_matrix = pd.DataFrame(positive_rho_matrix).reset_index(drop=True)
    positive_pval_matrix = pd.DataFrame(positive_pval_matrix).reset_index(drop=True)
    negative_rho_matrix = pd.DataFrame(negative_rho_matrix).reset_index(drop=True)
    negative_pval_matrix = pd.DataFrame(negative_pval_matrix).reset_index(drop=True)

    positive_rho_matrix.to_csv("output_csvs/lin_positive_ih_hm_rho.csv")
    positive_pval_matrix.to_csv("output_csvs/lin_positive_ih_hm_pval.csv")
    negative_rho_matrix.to_csv("output_csvs/lin_negative_ih_hm_rho.csv")
    negative_pval_matrix.to_csv("output_csvs/lin_negative_ih_hm_pval.csv")
