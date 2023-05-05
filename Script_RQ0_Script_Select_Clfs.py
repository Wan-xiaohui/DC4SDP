import pandas as pd
import time
from utils.helper import *
from src import instance_hardness
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sys import stdout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pickle
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from src.instance_hardness import default_models_dic
import scipy.cluster.hierarchy as sch


def cv_prediction(path_to_datasets, path_to_saved_file, models_dic):
    data_list, label_list, fname = load_data(path_to_datasets)

    for index, file in enumerate(fname):

        print('File:\t' + file + '...', flush=True)

        try:
            data = data_list[index]
            label = label_list[index]

            dataset = pd.DataFrame(np.column_stack((data, label)))

            X = dataset.iloc[:, :-1]
            y = dataset.iloc[:, -1]

            test_truths, test_preds = instance_hardness.repeated_cross_validation(X, y, models_dic)

            pkfile = open(path_to_saved_file + file + '.pickle', 'wb')

            pickle.dump(test_truths, pkfile)
            pickle.dump(test_preds, pkfile)

        except Exception as e:
            print('The following error occurred in case of the dataset {}: \n{}'.format(file, e))


def COD_scores(cv_results, default_models_dic):
    _, _, fname = load_data(path_to_datasets)
    n_models = len(default_models_dic)
    COD_matrix = np.zeros((n_models * (n_models - 1) // 2))
    pw_distances = {}
    print(sorted(default_models_dic.keys()))

    for index, file in enumerate(fname):
        pw_distances[file] = np.zeros((n_models, n_models))

    for index, file in enumerate(fname):
        for i in range(n_models):
            for j in range(n_models):
                for m in range(len(cv_results[file][1])):
                    for n in range(5):
                        if cv_results[file][1][m][sorted(default_models_dic.keys())[i]][n] != \
                                cv_results[file][1][m][sorted(default_models_dic.keys())[j]][n]:
                            pw_distances[file][i][j] += 1
                pw_distances[file][i][j] /= len(cv_results[file][1]) * 10

    q = lambda i, j, n: n * j - j * (j + 1) // 2 + i - 1 - j

    for i in range(1, n_models):
        for j in range(i):
            for _, file in enumerate(fname):
                COD_matrix[q(i, j, n_models)] += pw_distances[file][i][j]

            COD_matrix[q(i, j, n_models)] /= len(fname)

    return COD_matrix


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


if __name__ == '__main__':

    # Save the test results of cross-validation to the specified path
    path_to_datasets = 'datasets/'
    path_to_saved_file = 'dump/cv_results/'
    cv_prediction(path_to_datasets, path_to_saved_file, default_models_dic)

    # Read the test results of cross-validation
    # Calculate the COD scores of all models and classify and select the models through clustering methods
    _, _, fname = load_data('datasets/')
    cv_results = {}

    for index, file in enumerate(fname):

        pkfile = open('dump/cv_results/' + file + '.pickle', 'rb')
        test_truths = pickle.load(pkfile)
        test_preds = pickle.load(pkfile)
        cv_results[file] = [test_truths, test_preds]

    COD_scores = COD_scores(cv_results, default_models_dic)

    # Represent the hierarchical clustering results as a dendrogram and save as a png file
    dendrogram = sch.dendrogram(sch.linkage(COD_scores, method='ward'))
    plt.title('Dendrogram')
    plt.ylabel('Classifier Output Difference')
    plt.savefig('plot_dendrogram.pdf')
    plt.show()








