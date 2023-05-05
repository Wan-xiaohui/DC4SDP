from utils.helper import *
from src import instance_hardness
import numpy as np
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from src import CFS
import copy
import seaborn as sns
from scipy.stats import variation

measures_list = ['kDN', 'DS', 'DCP', 'TD_P', 'TD_U', 'CL', 'MV', 'CB', 'F1', 'N1', 'N2', 'LSC', 'LSR', 'H', 'U']


def calc_ih_measures(path_to_datasets, path_to_saved_csvs, measures_list,
                     is_normalized=False, is_resampling=False, is_feature_selection=False):

    data_list, label_list, fname = load_data(path_to_datasets)


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

            dataset_scores_df = instance_hardness.hm_scores(X, y, measures_list)

            check_for_nan = dataset_scores_df.isnull().values.any()

            if check_for_nan:
                print("NAN!")

            saved_csv_name = path_to_saved_csvs + '{}_ih_measures.csv'.format(file)
            dataset_scores_df.to_csv(saved_csv_name, header=True, index=False)

        except Exception as e:
            print('The following error occurred in case of the dataset {}: \n{}'.format(file, e))


def calc_ih_measures_correlation(path_to_datasets, path_to_saved_csvs):
    data_list, label_list, fname = load_data(path_to_datasets)

    for index, file in enumerate(fname):
        csv_file = path_to_saved_csvs + '{}_ih_measures.csv'.format(file)
        result_df = pd.read_csv(csv_file)

        if index == 0:
            total_result_df = copy.deepcopy(result_df)

        else:
            total_result_df = pd.concat([total_result_df, result_df], axis=0, ignore_index=True)

    # calculate the CV of each column in the dataframe
    cv = total_result_df.apply(lambda x: variation(x))
    avg = total_result_df.mean()
    std = total_result_df.std()
    considered_result_df = total_result_df.loc[:, total_result_df.std() > 0.05]
    corrMatrix = total_result_df.corr(method="spearman").round(3)
    # corrMatrix = considered_result_df.corr(method="spearman").round(3)

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
                       annot_kws={"size": 12}
                       )
    # fontsize can be adjusted to not be giant
    # hmap.axes.set_title("spearman correlation matrix for the hardness measures", fontsize=20)
    # labelsize can be adjusted to not be giant
    hmap.tick_params(labelsize=12)

    # saves plot output, change'C:/Users/bague/Downloads/cases_correlation.png' to whatever your directory should be and the new filename
    plt.savefig('output_plots/ih_measures_correlation.pdf', dpi=500)
    plt.savefig('output_plots/ih_measures_correlation.eps', dpi=500)


if __name__ == '__main__':

    path_to_datasets = 'datasets/'
    path_to_saved_csvs = 'output_csvs/hardness_measures/'

    calc_ih_measures(path_to_datasets, path_to_saved_csvs, measures_list)
    calc_ih_measures_correlation(path_to_datasets, path_to_saved_csvs)


