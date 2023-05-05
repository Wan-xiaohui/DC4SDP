from utils.helper import *
from src import instance_hardness
import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import matthews_corrcoef


path_to_datasets = 'datasets/'
path_to_saved_file = 'dump/prediction_results/'


def calc_performance(label_true, label_pred):
    MCC = matthews_corrcoef(label_true, label_pred)
    return MCC


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


def calc_dist_ih(ih_dic):
    dataset_names = ih_dic.keys()

    total_ih = []
    for dataset_name in dataset_names:
        total_ih.extend(ih_dic[dataset_name])

    total_ih = np.array(total_ih)
    p0 = len(np.where(total_ih == 0.0)[0])/len(total_ih) * 100
    p1 = len(np.where(total_ih == 1.0)[0])/len(total_ih) * 100

    p = []
    for i in range(10):
        p.append(len(np.where((total_ih < (i+1)*0.1) & (total_ih >= i*0.1))[0])/len(total_ih) * 100)
    p.append(p1)
    p_reverse = np.array(list(reversed(p)))

    p_cumsum = list(reversed(np.cumsum(p_reverse)))
    p_cumsum[0] -= p0
    p_cumsum.insert(0, p0)

    return np.array(p_cumsum)


def plot_boxplot_ih(ih_dic, file_name):
    dataset_names = list(ih_dic.keys())
    dataset_median = []
    for dataset_name in dataset_names:
        ih_dic[dataset_name] = np.array(ih_dic[dataset_name])
        dataset_median.append(np.median(ih_dic[dataset_name]))

    dataset_median = np.array(dataset_median)
    sort_index = np.argsort(dataset_median).tolist()

    ih_values = []
    for i in sort_index:
        ih_dic[dataset_names[i]] = np.array(ih_dic[dataset_names[i]])
        ih_values.extend(ih_dic[dataset_names[i]])

    dataset = []
    datasets_size = [len(ih_dic[dataset_names[i]]) for i in sort_index]

    for i in range(len(datasets_size)):
        for _ in range(datasets_size[i]):
            dataset.append(str(sort_index[i]+1))

    ih_df = pd.DataFrame({'instance hardness': ih_values, 'dataset item': dataset})
    plt.figure(figsize=(13, 5), dpi=500)
    sns.set(style="darkgrid", palette="icefire")

    sns.boxplot(x="dataset item",
                y="instance hardness",
                data=ih_df,
                showmeans=True,
                meanprops={"marker": "+",
                           "markeredgecolor": "black",
                           "markersize": "10"
                           }
                )
    # plt.show()
    plt.savefig("output_plots/"+file_name+".pdf")
    # print()


    # total_ih = positive_ih.append(negative_ih)
    # # # dist_ih = np.concatenate([dist_ih_ind_positive, dist_ih_ind_negative], axis=0)
    # # total_ih = np.vstack((dist_ih_ind_positive, dist_ih_ind_negative))
    # # # total_ih = pd.DataFrame(total_ih)
    # # # total_ih.columns = ['=0.0', '＞0.0', '≥0.1', '≥0.2', '≥0.3', '≥0.4', '≥0.5', '≥0.6', '≥0.7', '≥0.8', '≥0.9', '=1.0']
    # # # total_ih.index = pd.Series(["Defective", "Non-Defective"])
    # plt.figure(dpi=500)
    # sns.set(style="darkgrid")
    # ax = sns.barplot(x="interval", y="percentage", hue="class", data=total_ih)
    # # # ax = sns.histplot(total_ih, stat="percent", bins=10)
    # # ax = sns.barplot(x=column_names, y=total_ih[0, :])
    # # # ax = sns.barplot(x=column_names, y=total_ih[1, :])
    #
    # ax.bar_label(ax.containers[0], fmt='%.1f')
    # #
    # # ax = sns.histplot(total_ih_class, stat="percent", cumulative=True, bins=10)
    # # ax.bar_label(ax.containers[0], fmt='%.1f')
    #
    # plt.xlabel("")
    # ax.savefig("")
    # sns.set_palette("hls")
    # sns.set_style("darkgrid")
    # sns.ecdfplot(total_ih_ind, stat="proportion")
    # sns.histplot(total_ih_class, stat="percent", cumulative=True, bins=10)
    # sns.ecdfplot(total_ih_class, stat="proportion")
    # plt.show()
    # # plt.savefig("output_plots/dist_of_ih.pdf")
    # print()

    # return total_ih_ind, total_ih_class


if __name__ == '__main__':

    model_names = ['KNN', 'NB', 'CART', 'LR', 'SVM', 'MLP',
                   'Boosted_RS', 'Greedy_RL', 'RF', 'SVMBoosting', 'MLPBoosting']

    # Train models using different algorithms with 5*5-fold cross-validation
    # and collect the model's predicted outputs on the instances
    cv_prediction(path_to_datasets, path_to_saved_file, model_names)

    # Read the test results of all classifiers
    test_truth_dic, test_pred_dic, test_pred_prob_dic = get_prediction_results(path_to_datasets, path_to_saved_file)

    ih_ind_dic, ih_class_dic = calc_ih(model_names, test_truth_dic, test_pred_dic, test_pred_prob_dic)
    ih_ind_positive_dic, ih_ind_negative_dic, ih_class_positive_dic, ih_class_negative_dic = calc_partial_ih(
        model_names, test_truth_dic, test_pred_dic, test_pred_prob_dic)

    file_name = "total_ih_boxplot"
    plot_boxplot_ih(ih_ind_dic, file_name)

    file_name = "ih_positive_boxplot"
    plot_boxplot_ih(ih_ind_positive_dic, file_name)

    file_name = "ih_negative_boxplot"
    plot_boxplot_ih(ih_ind_negative_dic, file_name)





